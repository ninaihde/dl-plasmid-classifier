import click
import glob
import numpy as np
import os
import random
import re
import shutil
import torch

from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
from varname import nameof


def prepare_folders(original_data_path, train_percentage, val_percentage):
    # get all path names
    all_chr_paths = glob.glob(f'{original_data_path}/*/chr_fast5/*.fast5')
    all_plasmid_paths = glob.glob(f'{original_data_path}/*/plasmid_fast5/*.fast5')
    if not all_chr_paths or not all_plasmid_paths:
        raise ValueError('No fast5 files found!')

    # randomly split files according to chosen percentages
    chr_train, chr_val = split(all_chr_paths, train_percentage, val_percentage)
    plasmid_train, plasmid_val = split(all_plasmid_paths, train_percentage, val_percentage)
    test = [p for p in all_chr_paths + all_plasmid_paths if p not in chr_train + chr_val + plasmid_train + plasmid_val]

    # move data to new folders according to splitting
    for ds in [(chr_train, nameof(chr_train)), (chr_val, nameof(chr_val)),
               (plasmid_train, nameof(plasmid_train)), (plasmid_val, nameof(plasmid_val)),
               (test, nameof(test))]:
        move_files(original_data_path, ds)


def move_files(original_data_path, ds):
    # create folder if it does not exist yet
    folder_path = f'{original_data_path.split("/", 1)[0]}/prototype_{ds[1]}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for file_path in ds[0]:
        file_path = file_path.replace('\\', '/')  # TODO: remove line

        # add suffix to avoid overwriting files and to store labels -> batchname__runname__label
        file_path_splitted = re.split('[/.]+', file_path)
        unique_filename = f'{file_path_splitted[-2]}' \
                          f'__{file_path_splitted[-4]}' \
                          f'__{file_path_splitted[-3].split("_")[0]}.fast5'
        # copy file to new folder
        shutil.copyfile(file_path, f'{folder_path}/{unique_filename}')


def split(paths, train_percentage, val_percentage):
    train = random.sample(paths, k=int(len(paths) * train_percentage))
    val = random.sample([p for p in paths if p not in train], k=int(len(paths) * val_percentage))

    return train, val


def normalize(data_test, xi, outpath, pos=True):
    # normalize using z-score with median absolute deviation
    mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
    m = np.median(data_test, axis=1)
    data_test = ((data_test - np.expand_dims(m, axis=1)) * 1.0) / (1.4826 * np.expand_dims(mad, axis=1))

    # replace extreme signals (modified z-score larger than 3.5) with average of closest neighbors
    x = np.where(np.abs(data_test) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            data_test[x[0][i], x[1][i]] = data_test[x[0][i], x[1][i] + 1]
        elif x[1][i] == 2999:
            data_test[x[0][i], x[1][i]] = data_test[x[0][i], x[1][i] - 1]
        else:
            data_test[x[0][i], x[1][i]] = (data_test[x[0][i], x[1][i] - 1] + data_test[x[0][i], x[1][i] + 1]) / 2

    data_test = torch.tensor(data_test).float()
    if pos is True:
        torch.save(torch.tensor(data_test).float(), outpath + '/pos_' + str(xi) + '.pt')
    else:
        torch.save(torch.tensor(data_test).float(), outpath + '/neg_' + str(xi) + '.pt')


@click.command()
@click.option('--gtPos', '-gp', help='Ground truth list of positive read IDs')
@click.option('--gtNeg', '-gn', help='Ground truth list of negative read IDs')
@click.option('--inpath', '-i', help='The input fast5 directory path')
@click.option('--outpath', '-o', help='The output pytorch tensor directory path')
@click.option('--batch', '-b', default=10000, help='Batch size, default 10000')
@click.option('--cutoff', '-c', default=1500, help='Cutoff the first c signals')
def main(gtpos, gtneg, inpath, outpath, batch, cutoff):
    # read in pos and neg ground truth variables
    my_file_pos = open(gtpos, "r")
    posli = my_file_pos.readlines()
    my_file_pos.close()
    posli = [pi.split('\n')[0] for pi in posli]

    my_file_neg = open(gtneg, "r")
    negli = my_file_neg.readlines()
    my_file_neg.close()
    negli = [pi.split('\n')[0] for pi in negli]

    # make output folder
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print("##### posli and negli length")
    print(len(posli))
    print(len(negli))
    print()

    # split fast5 files
    arrneg = []
    arrpos = []
    pi = 0
    ni = 0

    for fileNM in glob.glob(inpath + '/*.fast5'):
        with get_fast5_file(fileNM, mode="r") as f5:
            print("##### file: " + fileNM)
            for read in f5.get_reads():
                raw_data = read.get_raw_data(scale=True)

                # only parse reads that are long enough
                if len(raw_data) >= (cutoff + 3000):
                    if read.read_id in posli:
                        pi += 1
                        arrpos.append(raw_data[cutoff:(cutoff + 3000)])
                        if (pi % batch == 0) and (pi != 0):
                            normalize(arrpos, pi, outpath, pos=True)
                            del arrpos
                            arrpos = []

                    if read.read_id in negli:
                        ni += 1
                        arrneg.append(raw_data[cutoff:(cutoff + 3000)])
                        if (ni % batch == 0) and (ni != 0):
                            normalize(arrneg, ni, outpath, pos=False)
                            del arrneg
                            arrneg = []


if __name__ == '__main__':
    main()
    #prepare_folders('data/prototype_original', 0.6, 0.2)  # -> 0.2 test
