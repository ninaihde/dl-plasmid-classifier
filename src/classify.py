import click
import csv
import glob
import numpy as np
import os
import time
import torch

from model import Bottleneck, ResNet
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def get_raw_data(file, reads, reads_ids, cutoff, seq_len, cut_after):
    with get_fast5_file(file, mode='r') as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data(scale=True)
            if len(raw_data) >= (cutoff + seq_len):
                if cut_after:
                    reads.append(raw_data[cutoff:])
                else:
                    reads.append(raw_data[cutoff:(cutoff + seq_len)])
                reads_ids.append(read.read_id)
    return reads, reads_ids


def normalize(reads, batch_idx):
    # normalize using z-score with median absolute deviation
    m = np.median(reads, axis=1)
    mad = stats.median_abs_deviation(reads, axis=1, scale='normal')
    reads = ((reads - np.expand_dims(m, axis=1)) * 1.0) / (1.4826 * np.expand_dims(mad, axis=1))

    # replace extreme signals (modified absolute z-score larger than 3.5) with average of closest neighbors
    # see Iglewicz and Hoaglin (https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf)
    # x[0] indicates read and x[1] signal in read
    x = np.where(np.abs(reads) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            reads[x[0][i], x[1][i]] = reads[x[0][i], x[1][i] + 1]
        elif x[1][i] == (reads[x[0][i]] - 1):
            reads[x[0][i], x[1][i]] = reads[x[0][i], x[1][i] - 1]
        else:
            reads[x[0][i], x[1][i]] = (reads[x[0][i], x[1][i] - 1] + reads[x[0][i], x[1][i] + 1]) / 2

    print(f'[Step 2] Done data normalization with batch {str(batch_idx)}')
    return torch.tensor(reads).float()


def process(reads, read_ids, batch_idx, bmodel, outpath, device):
    with torch.no_grad():
        data = reads.to(device)
        outputs = bmodel(data)
        with open(f'{outpath}/batch_{str(batch_idx)}.csv', 'w', newline='\n') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Read ID', 'Predicted Label'])
            for read_id, label_nr in zip(read_ids, outputs.max(dim=1).indices.int().data.cpu().numpy()):
                label = 'plasmid' if label_nr == 1 else 'chr'
                csv_writer.writerow([read_id, label])
        print(f'[Step 3] Done processing with batch {str(batch_idx)}')
        del outputs


@click.command()
@click.option('--model', '-m', help='path to pre-trained model', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='path to folder with input fast5 data', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='output result folder path', type=click.Path())
@click.option('--seq_len', '-s', default=3000, help='number of raw signals used per read')
@click.option('--cut_after', '-a', default=False,
              help='whether random sequence length per read is applied before or after normalization')
@click.option('--batch_size', '-b', default=1, help='batch size')  # test data is sorted by time, so should be representative
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
def main(model, inpath, outpath, seq_len, cut_after, batch_size, cutoff):
    start_time = time.time()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load pre-trained model
    bmodel = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device).eval()
    bmodel.load_state_dict(torch.load(model, map_location=device))
    print('[Step 0] Done loading model')

    reads = []
    read_ids = []
    batch_idx = 0
    for file in glob.glob(inpath + '/*.fast5'):
        # load pA signals and make all reads same-sized
        reads, read_ids = get_raw_data(file, reads, read_ids, cutoff, seq_len, cut_after)

        # ensure that cutoff has not removed all reads
        if len(reads) > 0:
            batch_idx += 1

            if batch_idx % batch_size == 0:
                print(f'[Step 1] Done loading data with batch {str(batch_idx)}, '
                      f'Getting {str(len(reads))} of sequences')
                reads = normalize(reads, batch_idx)
                if cut_after:
                    reads = [r[:seq_len] for r in reads]
                process(reads, read_ids, batch_idx, bmodel, outpath, device)
                print(f'[Step 4] Done with batch {str(batch_idx)}\n')

                del reads
                reads = []
                del read_ids
                read_ids = []

    print(f'[Step FINAL] --- {time.time() - start_time} seconds ---')


if __name__ == '__main__':
    main()
