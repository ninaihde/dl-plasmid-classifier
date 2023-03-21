"""
This script optimizes the decision threshold used while inference with respect to the balanced accuracy of the final
classification decision. The optimization is based on a subset (default: 30%) of the real data. For each maximum sequence
length, one version of the real tuning data is created and a respective optimal threshold is calculated.
"""

import click
import glob
import numpy as np
import ont_fast5_api.fast5_read as f5read
import os
import pandas as pd
import torch

from model import Bottleneck, ResNet
from numpy import random
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.multi_fast5 import MultiFast5File
from scipy import stats
from sklearn.metrics import roc_curve
from tqdm import tqdm


PLASMID_LABEL = 0


def cut_reads(in_dir, out_dir, cutoff, min_seq_len, max_seq_len, random_gen):
    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')

    ds_name = os.path.basename(out_dir)
    kept_reads = 0

    # create file for ground truth labels
    label_df = pd.DataFrame(columns=['Read ID', 'GT Label'])

    for input_file in tqdm(glob.glob(f'{in_dir}/*.fast5')):
        output_file = os.path.join(out_dir, os.path.basename(input_file))
        label = 'plasmid' if any(c in os.path.basename(input_file) for c in ['plasmid', 'pos']) else 'chr'

        with get_fast5_file(input_file, mode='r') as f5_old, MultiFast5File(output_file, mode='w') as f5_new:
            for i, read in enumerate(f5_old.get_reads()):
                # get random sequence length per read
                seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

                # only parse reads that are long enough
                if len(read.handle[read.raw_dataset_name]) >= (cutoff + seq_len):
                    kept_reads += 1
                    label_df = pd.concat(
                        [label_df, pd.DataFrame([{'Read ID': read.read_id, 'GT Label': label}])],
                        ignore_index=True)

                    # fill new fast5 file
                    read_name = f'read_{read.read_id}'
                    f5_new.handle.create_group(read_name)
                    output_group = f5_new.handle[read_name]
                    f5read.copy_attributes(read.handle.attrs, output_group)
                    for subgroup in read.handle:
                        if subgroup == read.raw_dataset_group_name:
                            raw_attrs = read.handle[read.raw_dataset_group_name].attrs
                            # remove cutoff and apply random sequence length
                            raw_data = read.handle[read.raw_dataset_name][cutoff:(cutoff + seq_len)]
                            output_read = f5_new.get_read(read.read_id)
                            output_read.add_raw_data(raw_data, raw_attrs)
                            new_attr = output_read.handle[read.raw_dataset_group_name].attrs
                            new_attr['duration'] = seq_len
                            continue
                        elif subgroup == 'channel_id':
                            output_group.copy(read.handle[subgroup], subgroup)
                            continue
                        else:
                            if read.run_id in f5_new.run_id_map:
                                # there may be a group to link to, but we must check it actually exists
                                hardlink_source = f'read_{f5_new.run_id_map[read.run_id]}/{subgroup}'
                                if hardlink_source in f5_new.handle:
                                    hardlink_dest = f'read_{read.read_id}/{subgroup}'
                                    f5_new.handle[hardlink_dest] = f5_new.handle[hardlink_source]
                                    continue
                            # if we couldn't hardlink to anything, we need to make the created group available for future reads
                            f5_new.run_id_map[read.run_id] = read.read_id
                        # if we haven't done a special-case copy, we can fall back on the default copy
                        output_group.copy(read.handle[subgroup], subgroup)

    # store ground truth labels of kept reads
    label_df.to_csv(f'{out_dir}/max{int(max_seq_len/1000)}_gt_{ds_name}_labels.csv', index=False)
    print(f'Number of reads in {out_dir}: {kept_reads}')


def append_read(read, reads, read_ids):
    reads.append(read.get_raw_data(scale=True))
    read_ids.append(read.read_id)

    return reads, read_ids


def normalize(data, consistency_correction=1.4826):
    extreme_signals = list()

    for r_i, read in enumerate(data):
        # normalize using z-score with median absolute deviation
        median = np.median(read)
        mad = stats.median_abs_deviation(read, scale='normal')
        data[r_i] = list((read - median) / (consistency_correction * mad))

        # get extreme signals (modified absolute z-score larger than 3.5)
        # see Iglewicz and Hoaglin (https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf)
        extreme_signals += [(r_i, s_i) for s_i, signal_is_extreme in enumerate(np.abs(data[r_i]) > 3.5)
                            if signal_is_extreme]

    # replace extreme signals with average of closest neighbors
    for read_idx, signal_idx in extreme_signals:
        if signal_idx == 0:
            data[read_idx][signal_idx] = data[read_idx][signal_idx + 1]
        elif signal_idx == (len(data[read_idx]) - 1):
            data[read_idx][signal_idx] = data[read_idx][signal_idx - 1]
        else:
            data[read_idx][signal_idx] = (data[read_idx][signal_idx - 1] + data[read_idx][signal_idx + 1]) / 2

    return data


def process(reads, read_ids, model, device):
    reads = torch.tensor(reads).float()

    with torch.no_grad():
        data = reads.to(device)
        outputs = model(data)
        sm = torch.nn.Softmax(dim=1)
        scores = sm(outputs)

        # get scores of target class (plasmids have label 0)
        plasmid_scores = scores[:, PLASMID_LABEL].cpu().numpy()
        results_per_batch = pd.DataFrame({'Read ID': read_ids, 'Score': plasmid_scores})
        del outputs

    return results_per_batch


def classify(in_dir, batch_size, max_seq_len, model, device):
    results = pd.DataFrame()
    reads = list()
    read_ids = list()
    n_reads = 0
    batch_idx = 0

    files = glob.glob(f'{in_dir}/*.fast5')
    for f_idx, file in enumerate(files):
        with get_fast5_file(file, mode='r') as f5:
            reads_to_process = f5.get_read_ids()
            for r_idx, read in enumerate(f5.get_reads()):
                reads, read_ids = append_read(read, reads, read_ids)
                n_reads += 1

                if (n_reads == batch_size) or ((f_idx == len(files) - 1) and (r_idx == len(reads_to_process) - 1)):
                    reads = normalize(reads)

                    # pad with zeros until maximum sequence length
                    reads = [r + [0] * (max_seq_len - len(r)) for r in reads]

                    # calculate and store scores
                    results_per_batch = process(reads, read_ids, model, device)
                    results = pd.concat([results, results_per_batch], ignore_index=True)
                    print(f'Classification of batch {batch_idx} completed.')

                    del reads
                    reads = []
                    del read_ids
                    read_ids = []
                    batch_idx += 1
                    n_reads = 0

    return results


def get_best_threshold(results, mx):
    fpr, tpr, thresholds = roc_curve(results['GT Label'].tolist(), results['Score'].tolist(), pos_label='plasmid')
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    print(f'Optimal decision threshold/s for maximum_sequence_length={mx}: {best_threshold}')


@click.command()
@click.option('--input_dir', '-i', type=click.Path(exists=True), required=True, help='folder path to real tune data')
@click.option('--trained_model', '-m', type=click.Path(exists=True), required=True, help='path to trained model')
@click.option('--batch_size', '-b', default=2000, help='number of reads per batch')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of signals per read')
@click.option('--max_seq_lens', '-max', multiple=True, default=[4000, 6000, 8000],
              help='maximum number of signals per read, defining several is possible')
def main(input_dir, trained_model, batch_size, random_seed, cutoff, min_seq_len, max_seq_lens):
    random_gen = random.default_rng(random_seed)

    # create three versions of tune data, one per maximum sequence length
    for mx in max_seq_lens:
        output_dir = f'{input_dir}_max{int(mx / 1000)}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cut_reads(input_dir, output_dir, cutoff, min_seq_len, mx, random_gen)

    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))

    # get best decision threshold per maximum sequence length
    for mx in max_seq_lens:
        tune_input = f'{input_dir}_max{int(mx / 1000)}'
        results = classify(tune_input, batch_size, mx, model, device)
        gt_labels = pd.read_csv(f'{tune_input}/max{int(mx/1000)}_gt_{os.path.basename(tune_input)}_labels.csv')
        results = pd.merge(results, gt_labels, left_on='Read ID', right_on='Read ID')
        get_best_threshold(results, mx)


if __name__ == '__main__':
    main()
