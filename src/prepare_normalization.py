"""
PREPROCESSING STEP 3/4
This script performs all steps that need to be done once on the simulated data. Therefore, it converts the simulated
single-fast5 files into compressed multi-fast5 files. Additionally, it splits the negative reads into train, validation
and test data. Finally, it removes all original simulation directories to reduce the amount of stored files.
"""

import click
import glob
import os
import shutil

from numpy import random
from ont_fast5_api.conversion_tools.single_to_multi_fast5 import batch_convert_single_to_multi


def merge_and_compress(sim_dir, class_type):
    print('Merge and compress simulated fast5 files...')
    output_dir = f'{sim_dir}_merged'
    for folder in glob.glob(f'{sim_dir}/{class_type}/*.fasta'):
        print(f'Folder: {folder}')
        input_dir = f'{folder}/fast5'
        if len(os.listdir(folder)) != 0:
            batch_convert_single_to_multi(input_path=input_dir, output_folder=output_dir,
                                          filename_base=class_type[:3], batch_size=5000, threads=32,
                                          target_compression='gzip', recursive=False, follow_symlinks=False)
    print(f'Number of files in {output_dir}: {len(glob.glob(f"{output_dir}/*.fast5"))}')


def split_randomly(paths, train_percentage, val_percentage, random_gen):
    train = random_gen.choice(paths, size=int(len(paths) * train_percentage), replace=False)
    val = random_gen.choice([p for p in paths if p not in train], size=int(len(paths) * val_percentage), replace=False)
    test = [p for p in paths if p not in train and p not in val]

    print(f'Training dataset:   {len(train)} / {len(paths)}')
    print(f'Validation dataset: {len(val)}   / {len(paths)}')
    print(f'Test dataset:       {len(test)}  / {len(paths)}')
    return train, val, test


def split_neg_reads(sim_neg, train_percentage, val_percentage, random_gen, train_sim_neg, val_sim_neg, test_sim_neg):
    print('Randomly splitting negative reads according to given percentages...')

    paths = glob.glob(f'{sim_neg}/*.fast5')
    train, val, test = split_randomly(paths, train_percentage, val_percentage, random_gen)

    for (ds_files, ds_dir) in [(train, train_sim_neg), (val, val_sim_neg), (test, test_sim_neg)]:
        if not os.path.exists(ds_dir):
            os.makedirs(ds_dir)

        for file in ds_files:
            shutil.copyfile(file, f'{ds_dir}/{os.path.basename(file)}')

        print(f'Reads successfully moved to {os.path.basename(ds_dir)}.')


def combine_folders(dir_out, dir_in_neg, dir_in_pos):
    print(f'Combining files into {dir_out}...')

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for file in glob.glob(f'{dir_in_neg}/*.fast5') + glob.glob(f'{dir_in_pos}/*.fast5'):
        shutil.copyfile(file, f'{dir_out}/{os.path.basename(file)}')


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
@click.option('--test_sim_neg', type=click.Path(), required=True,
              help='directory for simulated test data for negative class (.fast5)')
@click.option('--test_sim_pos', type=click.Path(exists=True), required=True,
              help='directory containing simulated test data for positive class (.fast5)')
@click.option('--random_seed', '-s', default=42, help='seed for random splitting')
@click.option('--train_pct', '-t', default=0.8, help='splitting percentage for negative training reads')
@click.option('--val_pct', '-v', default=0.1, help='splitting percentage for negative validation reads')
def main(sim_neg, train_sim_neg, train_sim_pos, val_sim_neg, val_sim_pos, test_sim, test_sim_neg, test_sim_pos,
         random_seed, train_pct, val_pct):
    # convert simulated single-fast5 files into compressed multi-fast5 files
    merge_and_compress(sim_neg, 'negative')
    merge_and_compress(train_sim_pos, 'positive')
    merge_and_compress(val_sim_pos, 'positive')
    merge_and_compress(test_sim_pos, 'positive')

    # split simulated reads of negative class and move simulated test data into one folder
    random_gen = random.default_rng(random_seed)
    split_neg_reads(f'{sim_neg}_merged', train_pct, val_pct, random_gen, train_sim_neg, val_sim_neg, test_sim_neg)
    combine_folders(test_sim, test_sim_neg, f'{test_sim_pos}_merged')

    # remove all original simulation directories to reduce amount of stored files
    # TODO: compress and backup not used directories in share-dir beforehand?
    shutil.rmtree(sim_neg)
    shutil.rmtree(train_sim_pos)
    shutil.rmtree(val_sim_pos)
    shutil.rmtree(test_sim_pos)


if __name__ == '__main__':
    main()
