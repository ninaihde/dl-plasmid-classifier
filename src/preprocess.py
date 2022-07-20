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
    folder_path = f'{outpath}/prototype_{ds[1]}'
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
        print(f'Files of dataset {ds[1]} already moved')


def normalize(data, batch_idx, outpath_ds, ds_name, max_seq_len, use_single_batch=False):
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
        elif x[1][i] == (max_seq_len - 1):
            data[x[0][i], x[1][i]] = data[x[0][i], x[1][i] - 1]
        else:
            data[x[0][i], x[1][i]] = (data[x[0][i], x[1][i] - 1] + data[x[0][i], x[1][i] + 1]) / 2

    # save as torch tensor (overwrites already existing file)
    data = torch.tensor(data).float()
    tensor_path = f'{outpath_ds}/{ds_name.split("_")[1]}_{ds_name.split("_")[2]}' \
                  f'{"" if use_single_batch else "_" + str(batch_idx)}.pt'
    torch.save(data, tensor_path)
    print(f'Torch tensor saved: {tensor_path}')


@click.command()
@click.option('--inpath', '-in',
              help='input directory path with folder per run, each run folder must contain a folder per class',
              type=click.Path(exists=True))
@click.option('--outpath', '-out',
              help='output directory path, folder per dataset with tensor files will be created',
              type=click.Path())
@click.option('--train_pct', '-tp', default=0.6, help='splitting percentage for training set')
@click.option('--val_pct', '-vp', default=0.2, help='splitting percentage for validation set')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of raw signals used per read')
@click.option('--max_seq_len', '-max', default=10000, help='maximum number of raw signals used per read')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--batch_size', '-b', default=10000, help='batch size, set to zero to use whole dataset size')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
def main(inpath, outpath, train_pct, val_pct, min_seq_len, max_seq_len, random_seed, batch_size, cutoff):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    random_gen = random.default_rng(random_seed)

    # split data into train, validation and test set for both classes + create folder for each dataset
    # assumes that data is already splitted correctly if folder exists
    prepare_folders(inpath, outpath, train_pct, val_pct, random_gen)

    # introduce boolean indicating whether dataset size should be used as batch size
    use_single_batch = False
    if batch_size == 0:
        use_single_batch = True

    # TODO: change '\\' to '/' & remove replace()
    for ds in [x.split('\\')[-1] for x in glob.glob(f'{outpath}/*')
               if x != inpath.replace('/', '\\') and x.split('\\')[-1].startswith('prototype')]:
        print(f'\nDataset: {ds}')

        if 'reads' in locals():
            del reads
        reads = list()

        batch_idx = 0

        # create file for ground truth labels
        if ds.endswith('test'):
            # ensure that already existing file is replaced
            if os.path.isfile(f'{inpath}/test_data_labels.csv'):
                os.remove(f'{inpath}/test_data_labels.csv')
            ids_file = open(f'{inpath}/test_data_labels.csv', 'w', newline='\n')
            csv_writer = csv.writer(ids_file)
            csv_writer.writerow(['Read ID', 'GT Label'])
        else:
            # ensure that already existing file is replaced
            if os.path.isfile(f'{inpath}/read_ids_{ds}.txt'):
                os.remove(f'{inpath}/read_ids_{ds}.txt')
            ids_file = open(f'{inpath}/read_ids_{ds}.txt', 'w')

        for file in glob.glob(f'{outpath}/{ds}/*.fast5'):
            print(f'File: {file}')
            label = re.split('[_.]+', file)[-2]

            with get_fast5_file(file, mode='r') as f5:
                for read in f5.get_reads():
                    # store ground truth labels (i.e., read ids per dataset or with labels)
                    if ds.endswith('test'):
                        csv_writer.writerow([read.read_id, label])
                        continue
                    else:
                        ids_file.write(f'{read.read_id}\n')

                    # get raw signals and convert to pA values
                    raw_data = read.get_raw_data(scale=True)

                    # generate random sequence length between min and max sequence length
                    seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

                    # only parse reads that are long enough
                    if len(raw_data) >= (cutoff + seq_len):
                        batch_idx += 1

                        # remove cutoff and signals after random sequence length
                        raw_data = raw_data[cutoff:(cutoff + seq_len)]
                        # pad read data with zeros until max_seq_len
                        raw_data = np.concatenate((raw_data, [0] * (max_seq_len - len(raw_data))), axis=None)

                        reads.append(raw_data)

                        # normalize if batch size should be used and is reached
                        if (not use_single_batch) and (batch_idx % batch_size == 0) and (batch_idx != 0):
                            normalize(reads, batch_idx, f'{outpath}/{ds}', ds, max_seq_len)
                            del reads
                            reads = list()

        # normalize all reads together (if dataset size should be used as batch size)
        if use_single_batch and not ds.endswith('test'):
            normalize(reads, batch_idx, f'{outpath}/{ds}', ds, max_seq_len, use_single_batch)
            del reads
            reads = list()

        print(f'Number of reads in dataset: {batch_idx}')
        ids_file.close()


if __name__ == '__main__':
    main()
