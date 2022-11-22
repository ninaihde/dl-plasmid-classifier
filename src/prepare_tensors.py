"""
PREPROCESSING STEP 3/3
First, this script splits the negative reads into train, validation and test data. Afterwards, it normalizes all train
and validation data using the z-score with the median absolute deviation. In addition, it performs cutting of the reads
to a randomly chosen sequence length and padding of the reads to a fixed length called max_seq_len. Finally, it saves
the train and validation data as torch tensors. For the testing datasets, only storing of the ground truth labels is
performed.
"""

import click
import glob
import numpy as np
import os
import pandas as pd
import re
import shutil
import time
import torch

from numpy import random
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def combine_folders(dir_out, dir_in_neg, dir_in_pos):
    print(f'Moving files into {dir_out}...')

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for file in glob.glob(f'{dir_in_neg}/*.fasta') + glob.glob(f'{dir_in_pos}/*.fasta'):
        shutil.copyfile(file, f'{dir_out}/{os.path.basename(file)}')


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
    tensor_path = f'{outpath_ds}/{outpath_ds.split("_")[-3]}_{outpath_ds.split("_")[-1]}' \
                  f'{"" if use_single_batch else "_" + str(batch_idx)}.pt'
    torch.save(data, tensor_path)
    print(f'Torch tensor saved: {tensor_path}')


@click.command()
@click.option('--test_real', '-test_r', type=click.Path(exists=True), required=True,
              help='directory containing real test data (.fast5)')
@click.option('--test_sim_neg', '-test_sn', type=click.Path(exists=True), required=True,
              help='directory containing simulated test data for negative class (.fast5)')
@click.option('--test_sim_pos', '-test_sp', type=click.Path(exists=True), required=True,
              help='directory containing simulated test data for positive class (.fast5)')
@click.option('--test_sim', '-test_s', type=click.Path(), required=True,
              help='directory for simulated test data of both classes (.fast5)')
@click.option('--test_sim_real_neg', '-test_srn', type=click.Path(exists=True), required=True,
              help='directory containing simulated real test data for negative class (.fast5)')
@click.option('--test_sim_real_pos', '-test_srp', type=click.Path(exists=True), required=True,
              help='directory containing simulated real test data for positive class (.fast5)')
@click.option('--test_sim_real', '-test_sr', type=click.Path(), required=True,
              help='directory for simulated real test data of both classes (.fast5)')
@click.option('--train_sim_neg', '-train_sn', type=click.Path(exists=True), required=True,
              help='directory containing simulated train data for negative class (.fast5)')
@click.option('--train_sim_pos', '-train_sp', type=click.Path(exists=True), required=True,
              help='directory containing simulated train data for positive class (.fast5)')
@click.option('--val_sim_neg', '-val_sn', type=click.Path(exists=True), required=True,
              help='directory containing simulated validation data for negative class (.fast5)')
@click.option('--val_sim_pos', '-val_sp', type=click.Path(exists=True), required=True,
              help='directory containing simulated validation data for positive class (.fast5)')
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help='directory for storing labels and read IDs')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of raw signals (after cutoff) used per read')
@click.option('--max_seq_len', '-max', default=8000, help='maximum number of raw signals (after cutoff) used per read')
@click.option('--cut_after', '-a', default=False,
              help='whether random sequence length per read of validation set is applied before or after normalization')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--batch_size', '-b', default=5000, help='batch size, set to zero to use whole dataset size')
def main(test_real, test_sim_neg, test_sim_pos, test_sim, test_sim_real_neg, test_sim_real_pos, test_sim_real,
         train_sim_neg, train_sim_pos, val_sim_neg, val_sim_pos, out_dir, cutoff, min_seq_len, max_seq_len, cut_after,
         random_seed, batch_size):
    start_time = time.time()

    combine_folders(test_sim, test_sim_neg, test_sim_pos)
    combine_folders(test_sim_real, test_sim_real_neg, test_sim_real_pos)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')

    random_gen = random.default_rng(random_seed)

    # introduce boolean indicating whether dataset size should be used as batch size
    use_single_batch = False
    if batch_size == 0:
        use_single_batch = True

    for ds_path in [test_real, test_sim, test_sim_real, train_sim_neg, train_sim_pos, val_sim_neg, val_sim_pos]:
        ds_name = os.path.basename(ds_path)
        print(f'\nDataset: {ds_name}')

        if 'reads' in locals():
            del reads
        reads = list()

        if 'seq_lengths' in locals():
            del seq_lengths
        seq_lengths = list()

        batch_idx = 0

        # create file for ground truth labels of validation and test datasets
        if 'val' in ds_name or 'test' in ds_name:
            label_df = pd.DataFrame(columns=['Read ID', 'GT Label'])

        for file_idx, file in enumerate(glob.glob(f'{ds_path}/*.fast5')):
            print(f'File: {file}')

            with get_fast5_file(file, mode='r') as f5:
                for read_idx, read in enumerate(f5.get_reads()):
                    # store ground truth labels for test dataset
                    if 'test' in ds_name:
                        label = re.split('[_.]+', file)[-2]
                        label_df = pd.concat(
                            [label_df, pd.DataFrame([{'Read ID': read.read_id, 'GT Label': label}])],
                            ignore_index=True)
                        continue

                    # get raw signals converted to pA values
                    raw_data = read.get_raw_data(scale=True)

                    # get random sequence length per read
                    seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

                    # only parse reads that are long enough
                    if len(raw_data) >= (cutoff + seq_len):
                        batch_idx += 1

                        if 'val' in ds_name:
                            label = 'plasmid' if ds_name.split('_')[2] == 'pos' else 'chr'
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

                            save_as_tensor(reads, ds_path, batch_idx)
                            del reads
                            reads = list()
                            del seq_lengths
                            seq_lengths = list()

        if 'test' not in ds_name:
            print(f'Number of kept reads in dataset: {batch_idx}')

            # normalize if single batch is used and all files are processed
            if use_single_batch and len(reads) > 0:
                reads = normalize(reads)

                for i in range(len(reads)):
                    # apply random sequence length
                    if cut_after and 'val' in ds_name:
                        reads[i] = reads[i][:seq_lengths[i]]

                    # pad with zeros until maximum sequence length
                    reads[i] += [0] * (max_seq_len - len(reads[i]))

                save_as_tensor(reads, ds_path, batch_idx, use_single_batch)
                del reads
                reads = list()
                del seq_lengths
                seq_lengths = list()

        # store ground truth labels
        if 'test' in ds_name:
            label_df.to_csv(f'{out_dir}/gt_{ds_name}_labels.csv', index=False)
        elif 'val' in ds_name:
            label_df.to_csv(f'{out_dir}/gt_val_{ds_name.split("_")[2]}_labels.csv', index=False)
            with open(f'{out_dir}/val_{ds_name.split("_")[2]}_read_ids.txt', 'w') as f:
                for r_id in label_df['Read ID'].tolist():
                    f.write(f'{str(r_id)}\n')

        print(f'Finished dataset. Runtime passed: {time.time() - start_time} seconds')

        # merging several produced .pt files of multiple-batch-modus of train resp. validation dataset
        if 'test' not in ds_name and not use_single_batch:
            print('Merging .pt files...')
            merged_tensors = torch.Tensor()
            for tensor_file in glob.glob(f'{ds_path}/*.pt'):
                current_tensor = torch.load(tensor_file)
                merged_tensors = torch.cat((merged_tensors, current_tensor))

            torch.save(merged_tensors, f'{ds_path}/{ds_name.split("_")[0]}_{ds_name.split("_")[2]}_merged.pt')
            print(f'Finished merging. Runtime passed: {time.time() - start_time} seconds')

    print(f'Overall runtime: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
