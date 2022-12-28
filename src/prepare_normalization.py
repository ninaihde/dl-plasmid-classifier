"""
PREPROCESSING STEP 3/4
This script performs all steps that need to be done once on the simulated data. Therefore, it converts the simulated
single-fast5 files into compressed multi-fast5 files. Additionally, it splits the negative reads into train, validation
and test data. Finally, it removes all original simulation directories to reduce the amount of stored files.
"""

import click
import glob
import os
import pandas as pd
import shutil

from numpy import random
from ont_fast5_api.conversion_tools.fast5_subset import Fast5Filter
from ont_fast5_api.conversion_tools.single_to_multi_fast5 import batch_convert_single_to_multi
from ont_fast5_api.fast5_interface import get_fast5_file


def merge_and_compress(input_dir, file_prefix, batch_size, threads):
    print(f'Merge and compress simulated fast5 files from {input_dir}...')
    output_dir = f'{input_dir}_merged'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_convert_single_to_multi(input_path=input_dir, output_folder=output_dir, filename_base=file_prefix,
                                  batch_size=batch_size, threads=threads, recursive=True, follow_symlinks=False,
                                  target_compression='gzip')

    print(f'Files of {input_dir} successfully merged into {output_dir}')


def extract_read_ids(input_dir):
    print('Extract IDs from negative reads...')
    read_ids = list()

    for file in glob.glob(f'{input_dir}/*.fast5'):
        with get_fast5_file(file, mode='r') as f5:
            read_ids += f5.get_read_ids()

    print(f'Extracted {len(read_ids)} read IDs.')
    return read_ids


def split_read_ids(read_ids, random_gen, train_percentage, val_percentage, input_dir):
    print('Randomly splitting negative read IDs according to given percentages...')
    train = random_gen.choice(read_ids, size=int(len(read_ids) * train_percentage), replace=False)
    val = random_gen.choice([p for p in read_ids if p not in train], size=int(len(read_ids) * val_percentage), replace=False)
    test = [p for p in read_ids if p not in train and p not in val]

    print(f'Training dataset:   {len(train)} / {len(read_ids)}')
    print(f'Validation dataset: {len(val)}   / {len(read_ids)}')
    print(f'Test dataset:       {len(test)}  / {len(read_ids)}')

    train_df = pd.DataFrame({'read_id': train})
    val_df = pd.DataFrame({'read_id': val})
    test_df = pd.DataFrame({'read_id': test})

    train_df.to_csv(f'{input_dir}/train_read_ids.csv', index=False, sep='\t')
    val_df.to_csv(f'{input_dir}/val_read_ids.csv', index=False, sep='\t')
    test_df.to_csv(f'{input_dir}/test_read_ids.csv', index=False, sep='\t')


def split_files(input_dir, output_dir, file_prefix, ds_type, batch_size, threads):
    print(f'Moving negative reads to folder {os.path.basename(output_dir)}...')
    Fast5Filter(input_folder=input_dir, output_folder=output_dir, filename_base=file_prefix,
                read_list_file=f'{input_dir}/{ds_type}_read_ids.csv', batch_size=batch_size, threads=threads,
                recursive=False, file_list_file=None, follow_symlinks=False, target_compression='gzip')


def split_neg_reads(input_dir, train_percentage, val_percentage, random_gen, train_sim_neg, val_sim_neg, test_sim,
                    batch_size, threads, file_prefix):
    read_ids = extract_read_ids(input_dir)
    split_read_ids(read_ids, random_gen, train_percentage, val_percentage, input_dir)

    split_files(input_dir, train_sim_neg, file_prefix, 'train', batch_size, threads)
    split_files(input_dir, val_sim_neg, file_prefix, 'val', batch_size, threads)
    split_files(input_dir, test_sim, file_prefix, 'test', batch_size, threads)


def move_files(output_dir, input_dir):
    print('Moving all simulated testing files into same folder...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in glob.glob(f'{input_dir}/*.fast5'):
        shutil.copyfile(file, f'{output_dir}/{os.path.basename(file)}')


@click.command()
@click.option('--sim_neg', type=click.Path(exists=True), required=True,
              help='directory containing simulated reads of negative class (.fast5)')
@click.option('--train_sim_neg', type=click.Path(), required=True,
              help='directory for simulated train data for negative class (.fast5)')
@click.option('--train_sim_pos', type=click.Path(exists=True), required=True,
              help='directory containing simulated train data for positive class (.fast5)')
@click.option('--val_sim_neg', type=click.Path(), required=True,
              help='directory for simulated validation data for negative class (.fast5)')
@click.option('--val_sim_pos', type=click.Path(exists=True), required=True,
              help='directory containing simulated validation data for positive class (.fast5)')
@click.option('--test_sim', type=click.Path(), required=True,
              help='directory for simulated test data of both classes (.fast5)')
@click.option('--test_sim_pos', type=click.Path(exists=True), required=True,
              help='directory containing simulated test data for positive class (.fast5)')
@click.option('--random_seed', '-s', default=42, help='seed for random splitting')
@click.option('--batch_size', '-b', default=8000, help='number of reads per batch for file generation')
@click.option('--threads', '-t', default=72, help='number of threads to use for fast5 merging and splitting')
@click.option('--train_pct', '-tp', default=0.8, help='splitting percentage for negative training reads')
@click.option('--val_pct', '-vp', default=0.1, help='splitting percentage for negative validation reads')
def main(sim_neg, train_sim_neg, train_sim_pos, val_sim_neg, val_sim_pos, test_sim, test_sim_pos, random_seed,
         batch_size, threads, train_pct, val_pct):

    # convert simulated single-fast5 files into compressed multi-fast5 files
    merge_and_compress(sim_neg, 'neg', batch_size, threads)
    merge_and_compress(train_sim_pos, 'pos', batch_size, threads)
    merge_and_compress(val_sim_pos, 'pos', batch_size, threads)
    merge_and_compress(test_sim_pos, 'pos', batch_size, threads)

    # split simulated reads of negative class and move simulated test data into one folder
    random_gen = random.default_rng(random_seed)
    split_neg_reads(f'{sim_neg}_merged', train_pct, val_pct, random_gen, train_sim_neg, val_sim_neg, test_sim,
                    batch_size, threads, 'neg')
    move_files(test_sim, f'{test_sim_pos}_merged')

    # remove all original simulation directories to reduce amount of stored files
    #shutil.rmtree(sim_neg)
    #shutil.rmtree(train_sim_pos)
    #shutil.rmtree(val_sim_pos)
    #shutil.rmtree(test_sim_pos)


if __name__ == '__main__':
    main()
