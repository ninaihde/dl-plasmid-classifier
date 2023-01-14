"""
This script reduces the number of normalized .pt files for the class with a larger number of files to the amount of
files that the other class has. The reduction is done by distributing all reads in the files to be removed evenly
among the files to be kept, i.e. the reads are appended at the end of the files to be kept. This reduction must be
executed for both the training and validation data to be used for training with the CustomDataLoader (which assumes the
same amount of files per dataset).
"""

import click
import glob
import os
import time
import torch

from tqdm import tqdm


def reduce_files(files, n_remaining_files):
    # extract and create new dir
    folder_name = os.path.dirname(files[0])
    new_dir = f'{os.path.dirname(folder_name)}/{os.path.basename(folder_name)}_ALIGNED'
    os.makedirs(new_dir)
    print(f'Aligned data will be stored in {new_dir}.')

    # extract files whose reads will be split across remaining files
    files_to_split = files[n_remaining_files:]
    remaining_files = files[:n_remaining_files]

    # extract how many reads should be added to each file
    n_reads = 0
    for f in files_to_split:
        n_reads += len(torch.load(f))
    n_reads_per_file = [n_reads // n_remaining_files + (1 if x < n_reads % n_remaining_files else 0)
                        for x in range(n_remaining_files)]

    # extend remaining tensor files
    reads_to_split = torch.Tensor()
    current_idx = 0
    for f_remain, nr in zip(tqdm(remaining_files), n_reads_per_file):
        for _ in range(current_idx, len(files_to_split)):
            if len(reads_to_split) >= nr:
                break
            else:
                reads_to_split = torch.cat((reads_to_split, torch.load(files_to_split[current_idx])))
                current_idx += 1

        merged_tensors = torch.cat((torch.load(f_remain), reads_to_split[:nr]))
        torch.save(merged_tensors, f'{new_dir}/{os.path.basename(f_remain)[:-3]}_aligned.pt')
        reads_to_split = reads_to_split[nr:]


@click.command()
@click.option('--pos_dir', '-p', help='folder with tensor files of positive class', required=True,
              type=click.Path(exists=True))
@click.option('--neg_dir', '-n', help='folder with tensor files of negative class', required=True,
              type=click.Path(exists=True))
def main(pos_dir, neg_dir):
    start_time = time.time()

    pos_files = [f for f in glob.glob(f'{pos_dir}/*.pt') if not f.endswith('tensors_merged.pt')]
    n_pos_files = len(pos_files)
    neg_files = [f for f in glob.glob(f'{neg_dir}/*.pt') if not f.endswith('tensors_merged.pt')]
    n_neg_files = len(neg_files)

    if n_pos_files > n_neg_files:
        print(f'Reducing amount of files for positive class...')
        reduce_files(pos_files, n_neg_files)
    elif n_pos_files < n_neg_files:
        print(f'Reducing amount of files for negative class...')
        reduce_files(neg_files, n_pos_files)
    elif n_pos_files == n_neg_files:
        print(f'Nothing changed because {pos_dir} and {neg_dir} contain the same amount of .pt files!')
        exit(0)

    print(f'Finished. Runtime: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
