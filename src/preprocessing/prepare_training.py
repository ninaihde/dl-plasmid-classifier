"""
PREPROCESSING STEP 4/4
This script normalizes all train and validation data using the z-score with the median absolute deviation. In
addition, it performs cutting of the reads to a randomly chosen sequence length and padding of the reads to a fixed
length called max_seq_len. Finally, it saves the train and validation data as torch tensors.
"""

import click
import glob
import numpy as np
import os
import pandas as pd
import time
import torch

from numpy import random
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def normalize(data):
    extreme_signals = list()

    for r_i, read in enumerate(data):
        # normalize using z-score with median absolute deviation
        median = np.median(read)
        mad = stats.median_abs_deviation(read, scale='normal')
        data[r_i] = list((read - median) / (1.4826 * mad))

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


def save_as_tensor(data, outpath_ds, batch_idx, use_single_batch=False):
    # Note: overwrites already existing file
    data = torch.tensor(data).float()
    tensor_path = f'{outpath_ds}/tensors{"" if use_single_batch else "_" + str(batch_idx)}.pt'
    torch.save(data, tensor_path)
    #print(f'Torch tensor saved: {tensor_path}')


@click.command()
@click.option('--train_sim_neg', type=click.Path(), required=True,
              help='directory for simulated train data for negative class (.fast5)')
@click.option('--train_sim_pos', type=click.Path(exists=True), required=True,
              help='directory containing simulated train data for positive class (.fast5)')
@click.option('--val_sim_neg', type=click.Path(), required=True,
              help='directory for simulated validation data for negative class (.fast5)')
@click.option('--val_sim_pos', type=click.Path(exists=True), required=True,
              help='directory containing simulated validation data for positive class (.fast5)')
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help='directory for storing labels, read IDs and tensor files')  # should start with prefix for filtering in evaluation
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of raw signals (after cutoff) used per read')
@click.option('--max_seq_len', '-max', default=8000, help='maximum number of raw signals (after cutoff) used per read')
@click.option('--cut_after', '-a', default=False,
              help='whether random sequence length per read of validation set is applied before or after normalization')
@click.option('--random_seed', '-s', default=42, help='seed for random sequence length generation')
@click.option('--batch_size', '-b', default=5000, help='amount of reads per batch, set to zero to use whole dataset size')
def main(train_sim_neg, train_sim_pos, val_sim_neg, val_sim_pos, out_dir, cutoff,
         min_seq_len, max_seq_len, cut_after, random_seed, batch_size):
    start_time = time.time()
    random_gen = random.default_rng(random_seed)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')

    # introduce boolean indicating whether dataset size should be used as batch size
    use_single_batch = False
    if batch_size == 0:
        use_single_batch = True

    for ds_path in [train_sim_neg, train_sim_pos, val_sim_neg, val_sim_pos]:
        ds_name = os.path.basename(ds_path)
        print(f'\nDataset: {ds_name}')

        if not os.path.exists(f'{out_dir}/{ds_name}'):
            os.makedirs(f'{out_dir}/{ds_name}')

        if 'reads' in locals():
            del reads
        reads = list()

        if 'seq_lengths' in locals():
            del seq_lengths
        seq_lengths = list()

        batch_idx = 0

        # create file for ground truth labels of validation datasets
        if 'val' in ds_name:
            label_df = pd.DataFrame(columns=['Read ID', 'GT Label'])

        for file in glob.glob(f'{ds_path}/*.fast5'):
            print(f'File: {file}')

            with get_fast5_file(file, mode='r') as f5:
                for read in f5.get_reads():
                    # get raw signals converted to pA values
                    raw_data = read.get_raw_data(scale=True)

                    # get random sequence length per read
                    seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

                    # only parse reads that are long enough
                    if len(raw_data) >= (cutoff + seq_len):
                        batch_idx += 1

                        # store ground truth labels for validation dataset
                        if 'val' in ds_name:
                            label = 'plasmid' if 'pos' in ds_name or 'plasmid' in ds_name else 'chr'
                            label_df = pd.concat(
                                [label_df, pd.DataFrame([{'Read ID': read.read_id, 'GT Label': label}])],
                                ignore_index=True)

                        if cut_after and 'val' in ds_name:
                            # only remove cutoff
                            raw_data = raw_data[cutoff:]
                            seq_lengths.append(seq_len)
                        else:
                            # remove cutoff and apply random sequence length
                            raw_data = raw_data[cutoff:(cutoff + seq_len)]

                        reads.append(raw_data)

                        # normalize if batch size is reached
                        if (not use_single_batch) and (batch_idx % batch_size == 0) and (batch_idx != 0):
                            reads = normalize(reads)

                            for i in range(len(reads)):
                                # apply random sequence length
                                if cut_after and 'val' in ds_name:
                                    reads[i] = reads[i][:seq_lengths[i]]

                                # pad with zeros until maximum sequence length
                                reads[i] += [0] * (max_seq_len - len(reads[i]))

                            save_as_tensor(reads, f'{out_dir}/{ds_name}', batch_idx)
                            del reads
                            reads = list()
                            del seq_lengths
                            seq_lengths = list()

        print(f'Number of kept reads in dataset: {batch_idx}')

        # normalize if single batch is used and all files are processed
        # or last batch did not reach batch size
        if (use_single_batch and len(reads) > 0) or (not use_single_batch and len(reads) > 0):
            reads = normalize(reads)

            for i in range(len(reads)):
                # apply random sequence length
                if cut_after and 'val' in ds_name:
                    reads[i] = reads[i][:seq_lengths[i]]

                # pad with zeros until maximum sequence length
                reads[i] += [0] * (max_seq_len - len(reads[i]))

            save_as_tensor(reads, f'{out_dir}/{ds_name}', batch_idx, use_single_batch)
            del reads
            reads = list()
            del seq_lengths
            seq_lengths = list()

        # store ground truth labels
        if 'val' in ds_name:
            label_df.to_csv(f'{out_dir}/gt_val_{"pos" if "pos" in ds_name or "plasmid" in ds_name else "neg"}_labels.csv', index=False)
            with open(f'{out_dir}/val_{"pos" if "pos" in ds_name or "plasmid" in ds_name else "neg"}_read_ids.txt', 'w') as f:
                for r_id in label_df['Read ID'].tolist():
                    f.write(f'{str(r_id)}\n')

        print(f'Finished dataset. Runtime passed: {time.time() - start_time} seconds')

    print(f'Overall runtime: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
