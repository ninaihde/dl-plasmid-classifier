"""
This inference procedure is an extended and adapted version of the one used in the SquiggleNet project, see
https://github.com/welch-lab/SquiggleNet/blob/master/inference.py. In addition to the original inference procedure, it
performs read padding like done for the train and validation data (see prepare_training.py) and uses a custom decision
threshold. In addition, it makes use of a certain number of reads per batch instead of a certain number of files per
batch.
"""

import click
import glob
import numpy as np
import os
import pandas as pd
import torch

from model import Bottleneck, ResNet
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def append_read(read, reads, read_ids):
    reads.append(read.get_raw_data(scale=True))
    read_ids.append(read.read_id)

    return reads, read_ids


def normalize(data, batch_idx, consistency_correction=1.4826):
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

    print(f'[Step 2] Done data normalization with batch {str(batch_idx)}')
    return data


def process(reads, read_ids, batch_idx, bmodel, outpath, device, threshold):
    # convert to torch tensors
    reads = torch.tensor(reads).float()

    with torch.no_grad():
        data = reads.to(device)
        outputs = bmodel(data)
        sm = torch.nn.Softmax(dim=1)
        scores = sm(outputs)

        # if score of target class > threshold, classify as plasmid
        # (opposite comparison because plasmids are indexed with zero)
        numbers = (scores[:, 0] <= threshold).int().data.cpu().numpy()
        labels = ['plasmid' if nr == 0 else 'chr' for nr in numbers]
        results = pd.DataFrame({'Read ID': read_ids, 'Predicted Label': labels})
        results.to_csv(f'{outpath}/batch_{str(batch_idx)}.csv', index=False)

        print(f'[Step 3] Done processing of batch {str(batch_idx)}')
        del outputs


@click.command()
@click.option('--model', '-m', help='input path to trained model', type=click.Path(exists=True), required=True)
@click.option('--inpath', '-i', help='input path to fast5 data', type=click.Path(exists=True), required=True)
# "max{max_seq_len}_{#epochs}epochs_{dataset}_{criterion}", e.g. max4_15epochs_sim_acc
@click.option('--outpath', '-o', help='output path for results', type=click.Path(), required=True)
@click.option('--max_seq_len', '-max', default=4000, help='maximum number of raw signals (after cutoff) used per read')
@click.option('--batch_size', '-b', default=1000, help='number of reads per batch')
@click.option('--threshold', '-s', default=0.5, help='threshold for final classification decision')
@click.option('--threads', '-t', default=32, help='number of threads to use for normalization')
def main(model, inpath, outpath, max_seq_len, batch_size, threshold, threads):
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load trained model
    bmodel = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device).eval()
    bmodel.load_state_dict(torch.load(model, map_location=device))
    print('[Step 0] Done loading model')

    reads, read_ids = list(), list()
    n_reads, batch_idx = 0, 0

    files = glob.glob(f'{inpath}/*.fast5')
    for f_idx, file in enumerate(files):
        with get_fast5_file(file, mode='r') as f5:
            reads_to_process = f5.get_read_ids()
            for r_idx, read in enumerate(f5.get_reads()):
                reads, read_ids = append_read(read, reads, read_ids)
                n_reads += 1

                if (n_reads == batch_size) or ((f_idx == len(files) - 1) and (r_idx == len(reads_to_process) - 1)):
                    print(f'[Step 1] Done loading data until batch {str(batch_idx)}')
                    reads = normalize(reads, batch_idx, threads)

                    # pad with zeros until maximum sequence length
                    reads = [r + [0] * (max_seq_len - len(r)) for r in reads]

                    process(reads, read_ids, batch_idx, bmodel, outpath, device, threshold)
                    print(f'[Step 4] Done with batch {str(batch_idx)}\n')

                    del reads
                    reads = []
                    del read_ids
                    read_ids = []
                    batch_idx += 1
                    n_reads = 0

    print(f'Classification of {batch_idx} batches finished.\n')


if __name__ == '__main__':
    main()
