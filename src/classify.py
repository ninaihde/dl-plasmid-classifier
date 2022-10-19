import click
import csv
import glob
import numpy as np
import os
import time
import torch

from model import Bottleneck, ResNet
from numpy import random
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def get_raw_data(file, reads, reads_ids, seq_lengths, cutoff, random_gen, min_seq_len, max_seq_len, cut_after):
    with get_fast5_file(file, mode='r') as f5:
        for read in f5.get_reads():
            # get raw and scaled read
            raw_data = read.get_raw_data(scale=True)

            # get random sequence length per read
            seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

            if len(raw_data) >= (cutoff + seq_len):
                reads_ids.append(read.read_id)

                if cut_after:
                    reads.append(raw_data[cutoff:])
                    seq_lengths.append(seq_len)
                else:
                    reads.append(raw_data[cutoff:(cutoff + seq_len)])

    return reads, reads_ids, seq_lengths


def normalize(data, batch_idx):
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

    print(f'[Step 2] Done data normalization with batch {str(batch_idx)}')
    return data


def process(reads, read_ids, batch_idx, bmodel, outpath, device):
    # convert to torch tensors
    reads = torch.tensor(reads).float()

    with torch.no_grad():
        data = reads.to(device)
        outputs = bmodel(data)
        with open(f'{outpath}/batch_{str(batch_idx)}.csv', 'w', newline='\n') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Read ID', 'Predicted Label'])
            for read_id, label_nr in zip(read_ids, outputs.max(dim=1).indices.int().data.cpu().numpy()):
                label = 'plasmid' if label_nr == 0 else 'chr'
                csv_writer.writerow([read_id, label])
        print(f'[Step 3] Done processing with batch {str(batch_idx)}')
        del outputs


@click.command()
@click.option('--model', '-m', help='path to pre-trained model', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='path to folder with input fast5 data', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='output result folder path', type=click.Path())
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of raw signals (after cutoff) used per read')
@click.option('--max_seq_len', '-max', default=4000, help='maximum number of raw signals (after cutoff) used per read')
@click.option('--cut_after', '-a', default=False,
              help='whether random sequence length per read is applied before or after normalization')
@click.option('--batch_size', '-b', default=1, help='batch size')  # test data is sorted by time, so 1 should be representative
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
def main(model, inpath, outpath, min_seq_len, max_seq_len, cut_after, batch_size, random_seed, cutoff):
    start_time = time.time()

    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')

    random_gen = random.default_rng(random_seed)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load pre-trained model
    bmodel = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device).eval()
    bmodel.load_state_dict(torch.load(model, map_location=device))
    print('[Step 0] Done loading model')

    reads = list()
    read_ids = list()
    seq_lengths = list()
    batch_idx = 0

    for file in glob.glob(inpath + '/*.fast5'):
        reads, read_ids, seq_lengths = \
            get_raw_data(file, reads, read_ids, seq_lengths, cutoff, random_gen, min_seq_len, max_seq_len, cut_after)

        # ensure that cutoff has not removed all reads
        if len(reads) > 0:
            batch_idx += 1

            if batch_idx % batch_size == 0:
                print(f'[Step 1] Done loading data until batch {str(batch_idx)}, '
                      f'Getting {str(len(reads))} of sequences')

                reads = normalize(reads, batch_idx)

                for i, r in enumerate(reads):
                    if cut_after:
                        # cut to random sequence length
                        reads[i] = reads[i][:seq_lengths[i]]

                    # pad with zeros until maximum sequence length
                    reads[i] += [0] * (max_seq_len - len(reads[i]))

                process(reads, read_ids, batch_idx, bmodel, outpath, device)
                print(f'[Step 4] Done with batch {str(batch_idx)}\n')

                del reads
                reads = []
                del read_ids
                read_ids = []
                del seq_lengths
                seq_lengths = []

    print(f'[Step FINAL] --- {time.time() - start_time} seconds ---')


if __name__ == '__main__':
    main()
