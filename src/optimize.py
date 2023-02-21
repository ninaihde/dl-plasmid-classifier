"""
This script optimizes the decision threshold used while inference with respect to the balanced accuracy of the final
classification decision. The optimization is based on a subset (default: 30%) of the original real testing data which we
call real train data.
"""

import click
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import torch

from model import Bottleneck, ResNet
from numpy import random
from ont_fast5_api.conversion_tools.fast5_subset import Fast5Filter
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
from sklearn.metrics import roc_curve, auc


PLASMID_LABEL = 0


# chose 5 plasmids related to Campylobacter coli, Campylobacter jejuni, Leptospira interrogans & Conchiformibius steedae
REAL_TRAIN_PLASMIDS = ['20220321_1207_MN24598_FAR91003', '20220419_1137_MN24598_FAR92750', 'l_interrogans', 'c_steedae']


def move_chromosomes_by_ids(chr_ids, input_dir, output_dir, batch_size, threads):
    df = pd.DataFrame({'read_id': chr_ids})
    df.to_csv(f'{output_dir}/{os.path.basename(output_dir)}_chromosomes.csv', index=False, sep='\t')

    extractor = Fast5Filter(input_folder=input_dir, output_folder=output_dir, filename_base='chr__',
                            read_list_file=f'{output_dir}/{os.path.basename(output_dir)}_chromosomes.csv',
                            batch_size=batch_size, threads=threads, recursive=False, follow_symlinks=False,
                            file_list_file=None)
    extractor.run_batch()


def move_chromosomes(random_gen, chr_gt, percentage, input_dir, output_train, output_test, batch_size, threads):
    # move real train data to new folder
    chr_ids = chr_gt['Read ID'].tolist()
    train_chr_ids = random_gen.choice(chr_ids, size=int(len(chr_ids) * percentage), replace=False)
    move_chromosomes_by_ids(train_chr_ids, input_dir, output_train, batch_size, threads)

    # move real test data to new folder
    test_df = chr_gt[~chr_gt['Read ID'].isin(train_chr_ids)]
    test_chr_ids = test_df['Read ID'].tolist()
    move_chromosomes_by_ids(test_chr_ids, input_dir, output_test, batch_size, threads)

    return train_chr_ids


def move_plasmids(input_dir, output_train, output_test):
    for plasmid_file in glob.glob(f'{input_dir}/*plasmid*.fast5'):
        if os.path.basename(plasmid_file).startswith(tuple(REAL_TRAIN_PLASMIDS)):
            shutil.copyfile(plasmid_file, f'{output_train}/{os.path.basename(plasmid_file)}')
        else:
            shutil.copyfile(plasmid_file, f'{output_test}/{os.path.basename(plasmid_file)}')


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


def get_best_threshold(results):
    fpr, tpr, thresholds = roc_curve(results['GT Label'].tolist(), results['Score'].tolist(), pos_label='plasmid')
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    print(f'Optimal decision threshold/s: {best_threshold}')


@click.command()
@click.option('--input_dir', '-i', type=click.Path(exists=True), required=True, help='folder path to original real test data')
@click.option('--test_dir', '-test', type=click.Path(), required=True, help='folder path to new real test data')
@click.option('--train_dir', '-train', type=click.Path(), required=True, help='folder path to real train data')
@click.option('--trained_model', '-m', type=click.Path(exists=True), required=True, help='path to trained model')
@click.option('--labels', '-l', type=click.Path(exists=True), required=True,
              help='file containing read IDs and ground truth labels of original real data')
@click.option('--random_seed', '-s', default=42, help='seed for selection of random chromosome read IDs')
@click.option('--percentage', '-p', default=0.3, help='percentage of reads to select for real training dataset')
@click.option('--batch_size_extract', '-be', default=10000, help='number of reads per batch for FAST5 file creation')
@click.option('--batch_size_classify', '-bc', default=2000, help='number of reads per batch for classification')
@click.option('--threads', '-t', default=32, help='number of threads to use for fast5 extraction and normalization')
@click.option('--max_seq_len', '-max', default=4, help='maximum number of raw signals used per read (in k)')
def main(input_dir, test_dir, train_dir, trained_model, labels, random_seed, percentage, batch_size_extract,
         batch_size_classify, threads, max_seq_len):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    random_gen = random.default_rng(random_seed)
    gt = pd.read_csv(labels)
    chr_gt = gt[gt['GT Label'] == 'chr']

    # split original real test folder
    train_chr_ids = move_chromosomes(random_gen, chr_gt, percentage, input_dir, train_dir, test_dir, batch_size_extract,
                                     threads)
    move_plasmids(input_dir, train_dir, test_dir)
    
    # store ground truth data of new real train folder
    gt_train = pd.DataFrame(columns=['Read ID', 'GT Label'])
    for file in glob.glob(f'{train_dir}/*plasmid*.fast5'):
        with get_fast5_file(file, mode='r') as f5:
            read_ids = f5.get_read_ids()
            gt_train = pd.concat(
                [gt_train, pd.DataFrame({'Read ID': read_ids, 'GT Label': ['plasmid'] * len(read_ids)})],
                ignore_index=True)
    gt_train = pd.concat([gt_train, pd.DataFrame({'Read ID': train_chr_ids, 'GT Label': ['chr'] * len(train_chr_ids)})],
                         ignore_index=True)
    gt_train.to_csv(f'{train_dir}/max{max_seq_len}_gt_train_real_labels.csv', index=False)

    # store ground truth data of new real test folder
    gt_test = pd.concat([gt, gt_train]).drop_duplicates(keep=False)
    gt_test.to_csv(f'{test_dir}/max{max_seq_len}_gt_test_real_labels_reduced.csv', index=False)
    del gt_test

    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))

    # extract plasmid scores & binary ground truth labels
    results = classify(train_dir, batch_size_classify, max_seq_len * 1000, model, device)
    results = pd.merge(results, gt_train, left_on='Read ID', right_on='Read ID')

    # print best threshold
    get_best_threshold(results)


if __name__ == '__main__':
    main()
