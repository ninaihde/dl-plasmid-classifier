import click
import csv
import glob
import numpy as np
import os
import re
import shutil
import torch

from numpy import random
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
from varname import nameof


def prepare_folders(inpath, outpath, train_percentage, val_percentage, random_gen):
    # get all path names
    chr_paths = glob.glob(f'{inpath}/*/chr_fast5/*.fast5')
    if not chr_paths:
        raise FileNotFoundError('No fast5 files containing bacterial chromosomes found!')

    plasmid_paths = glob.glob(f'{inpath}/*/plasmid_fast5/*.fast5')
    if not plasmid_paths:
        raise FileNotFoundError('No fast5 files containing plasmids found!')

    # randomly split files according to chosen percentages
    chr_train, chr_val = split(chr_paths, train_percentage, val_percentage, random_gen)
    plasmid_train, plasmid_val = split(plasmid_paths, train_percentage, val_percentage, random_gen)
    test = [p for p in chr_paths + plasmid_paths
            if p not in np.concatenate((chr_train, chr_val, plasmid_train, plasmid_val), axis=None)]

    # move data to new folders according to splitting
    for ds in [(chr_train, nameof(chr_train)), (chr_val, nameof(chr_val)),
               (plasmid_train, nameof(plasmid_train)), (plasmid_val, nameof(plasmid_val)),
               (test, nameof(test))]:
        move_files(outpath, ds)


def split(paths, train_percentage, val_percentage, random_gen):
    train = random_gen.choice(paths, size=int(len(paths) * train_percentage), replace=False)
    val = random_gen.choice([p for p in paths if p not in train], size=int(len(paths) * val_percentage), replace=False)

    # catch if dataset splitting percentages created empty folder
    label = paths[0].split('\\')[-2].split('_')[0]  # TODO: change '\\' to '/'
    if len(train) == 0:
        raise ValueError(f'No fast5 files selected for dataset {label}_train! Please adjust the splitting percentages.')
    if len(val) == 0:
        raise ValueError(f'No fast5 files selected for dataset {label}_val! Please adjust the splitting percentages.')

    return train, val


def move_files(outpath, ds):
    # create folder if it does not exist yet
    folder_path = f'{outpath}/prepared_{ds[1]}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

        for file_path in ds[0]:
            file_path = file_path.replace('\\', '/')  # TODO: remove line

            # add suffix to avoid overwriting files and to store labels for test data -> runname__batchname__label
            file_path_splitted = re.split('[/.]+', file_path)
            unique_filename = f'{file_path_splitted[-4]}' \
                              f'__{file_path_splitted[-2]}' \
                              f'__{file_path_splitted[-3].split("_")[0]}.fast5'
            # copy file to new folder
            shutil.copyfile(file_path, f'{folder_path}/{unique_filename}')

        print(f'Files of dataset {ds[1]} successfully moved')
    else:
        print(f'Files of dataset {ds[1]} were already moved')


def normalize(data, longest_max_seq_len):
    data = np.array(data)

    # normalize using z-score with median absolute deviation
    mad = stats.median_abs_deviation(data, axis=1, scale='normal')
    m = np.median(data, axis=1)
    data = ((data - np.expand_dims(m, axis=1)) * 1.0) / (1.4826 * np.expand_dims(mad, axis=1))

    # replace extreme signals (modified absolute z-score larger than 3.5) with average of closest neighbors
    # see Iglewicz and Hoaglin (https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf)
    # x[0] indicates read and x[1] signal in read
    x = np.where(np.abs(data) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            data[x[0][i], x[1][i]] = data[x[0][i], x[1][i] + 1]
        elif x[1][i] == (longest_max_seq_len - 1):
            data[x[0][i], x[1][i]] = data[x[0][i], x[1][i] - 1]
        else:
            data[x[0][i], x[1][i]] = (data[x[0][i], x[1][i] - 1] + data[x[0][i], x[1][i] + 1]) / 2

    return [list(r) for r in data]


def save_as_tensor(data, outpath_ds, batch_idx, use_single_batch=False):
    # Note: overwrites already existing file
    data = torch.tensor(data).float()
    tensor_path = f'{outpath_ds}/{outpath_ds.split("_")[-2]}_{outpath_ds.split("_")[-1]}' \
                  f'{"" if use_single_batch else "_" + str(batch_idx)}.pt'
    torch.save(data, tensor_path)
    print(f'Torch tensor saved: {tensor_path}')


@click.command()
@click.option('--inpath', '-i', type=click.Path(exists=True),
              help='input directory path with folder per run, each run folder must contain a folder per class')
@click.option('--outpath', '-o', type=click.Path(),
              help='output directory path, folder per dataset with tensor files will be created')
@click.option('--train_pct', '-t', default=0.8, help='splitting percentage for training set')
@click.option('--val_pct', '-v', default=0.1, help='splitting percentage for validation set')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of raw signals (after cutoff) used per read')
@click.option('--max_seq_len', '-max', default=8000, help='maximum number of raw signals (after cutoff) used per read')
@click.option('--longest_max_seq_len', '-lmax', default=8000,
              help='longest maximum number of raw signals (after cutoff) used per read in all compared runs. Set to '
                   'max_seq_len if only one run will be executed.')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--batch_size', '-b', default=10000, help='batch size, set to zero to use whole dataset size')
def main(inpath, outpath, train_pct, val_pct, cutoff, min_seq_len, max_seq_len, longest_max_seq_len, random_seed,
         batch_size):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # check script arguments
    if (train_pct <= 0.0) or (val_pct <= 0.0):
        raise ValueError('Chosen training and/or validation percentage must be bigger than 0.0!')
    if train_pct + val_pct >= 1.0:
        raise ValueError('In total, chosen training and validation percentage must be smaller than 1.0! Please adjust '
                         'these percentages to allow for a valid testing percentage.')
    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')
    if max_seq_len > longest_max_seq_len:
        raise ValueError('The maximum sequence length must be smaller than or equal to the longest maximum sequence '
                         'length!')

    random_gen = random.default_rng(random_seed)

    # split data into train, validation and test set for both classes + create folder for each dataset
    # assumes that data is already splitted correctly if folder exists
    prepare_folders(inpath, outpath, train_pct, val_pct, random_gen)

    # introduce boolean indicating whether dataset size should be used as batch size
    use_single_batch = False
    if batch_size == 0:
        use_single_batch = True

    # TODO: change '\\' to '/'
    for ds_name in [x.split('\\')[-1] for x in glob.glob(f'{outpath}/*') if x.split('\\')[-1].startswith('prepared')]:
        print(f'\nDataset: {ds_name}')

        if 'reads' in locals():
            del reads
        reads = list()

        batch_idx = 0

        # create file for ground truth labels of test dataset
        if ds_name.endswith('test'):
            # ensure that already existing file is replaced
            if os.path.isfile(f'{outpath}/gt_test_labels.csv'):
                os.remove(f'{outpath}/gt_test_labels.csv')
            ids_file = open(f'{outpath}/gt_test_labels.csv', 'w', newline='\n')
            csv_writer = csv.writer(ids_file)
            csv_writer.writerow(['Read ID', 'GT Label'])

        for file_idx, file in enumerate(glob.glob(f'{outpath}/{ds_name}/*.fast5')):
            print(f'File: {file}')
            label = re.split('[_.]+', file)[-2]

            with get_fast5_file(file, mode='r') as f5:
                for read_idx, read in enumerate(f5.get_reads()):
                    # store ground truth labels for test dataset
                    if ds_name.endswith('test'):
                        csv_writer.writerow([read.read_id, label])
                        continue

                    # get raw signals and convert to pA values
                    raw_data = read.get_raw_data(scale=True)

                    # only parse reads that are long enough
                    if len(raw_data) >= (cutoff + longest_max_seq_len):
                        batch_idx += 1

                        # remove cutoff and keep signals up to longest maximum sequence length
                        raw_data = raw_data[cutoff:(cutoff + longest_max_seq_len)]

                        reads.append(raw_data)

                        # normalize if all files are processed (single batch) or batch size is reached
                        all_files_processed = (file_idx == len(glob.glob(f'{outpath}/{ds_name}/*.fast5')) - 1
                                               and read_idx == len(f5.get_read_ids()) - 1)
                        if (use_single_batch and all_files_processed) \
                                or (not use_single_batch and (batch_idx % batch_size == 0) and (batch_idx != 0)):
                            reads = normalize(reads, longest_max_seq_len)

                            for i in range(len(reads)):
                                # apply maximum sequence length
                                reads[i] = reads[i][:max_seq_len]

                                # generate random suffix start between min and max sequence length (both inclusive)
                                suffix_start = random_gen.integers(min_seq_len, max_seq_len + 1)

                                # substitute suffix with zeros
                                reads[i][suffix_start:len(reads[i])] = [0] * (len(reads[i]) - suffix_start)

                            save_as_tensor(reads, f'{outpath}/{ds_name}', batch_idx, use_single_batch)
                            del reads
                            reads = list()

        if not ds_name.endswith('test'):
            print(f'Number of kept reads in dataset: {batch_idx}')


if __name__ == '__main__':
    main()
