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


########################
##### Normalization ####
########################
def normalize(data_test, batch_idx, max_seq_len):
    mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
    m = np.median(data_test, axis=1)
    data_test = ((data_test - np.expand_dims(m, axis=1)) * 1.0) / (1.4826 * np.expand_dims(mad, axis=1))

    x = np.where(np.abs(data_test) > 3.5)
    for i in range(x[0].shape[0]):
        if x[1][i] == 0:
            data_test[x[0][i], x[1][i]] = data_test[x[0][i], x[1][i] + 1]
        elif x[1][i] == (max_seq_len - 1):
            data_test[x[0][i], x[1][i]] = data_test[x[0][i], x[1][i] - 1]
        else:
            data_test[x[0][i], x[1][i]] = (data_test[x[0][i], x[1][i] - 1] + data_test[x[0][i], x[1][i] + 1]) / 2

    data_test = torch.tensor(data_test).float()

    print(f'[Step 2]$$$$$$$$$$ Done data normalization with batch {str(batch_idx)}')
    return data_test


########################
####### Run Test #######
########################
def process(data_test, data_name, batch_idx, bmodel, outpath, device):
    with torch.no_grad():
        data = data_test.to(device)
        outputs = bmodel(data)
        with open(f'{outpath}/batch_{str(batch_idx)}.csv', 'w', newline='\n') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Read ID', 'Predicted Label'])
            for read_id, label_nr in zip(data_name, outputs.max(dim=1).indices.int().data.cpu().numpy()):
                label = 'plasmid' if label_nr == 1 else 'chr'
                csv_writer.writerow([read_id, label])
        print(f'[Step 3]$$$$$$$$$$ Done processing with batch {str(batch_idx)}')
        del outputs


########################
#### Load the data #####
########################
def get_raw_data(file, data_test, data_name, cutoff, max_seq_len):
    with get_fast5_file(file, mode='r') as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data(scale=True)
            if len(raw_data) >= (cutoff + max_seq_len):
                data_test.append(raw_data[cutoff:(cutoff + max_seq_len)])
                data_name.append(read.read_id)
    return data_test, data_name


@click.command()
@click.option('--model', '-m', help='path to pre-trained model', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='path to folder with input fast5 data', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='output result folder path', type=click.Path())
@click.option('--max_seq_len', '-s', default=3000, help='number of raw signals used per read')
@click.option('--batch', '-b', default=1, help='batch size')  # test data is sorted by time, so should be representative
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
def main(model, inpath, outpath, max_seq_len, batch, cutoff):
    start_time = time.time()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load pre-trained model
    layers = [2, 2, 2, 2]
    bmodel = ResNet(Bottleneck, layers).to(device).eval()
    bmodel.load_state_dict(torch.load(model, map_location=device))
    print('[Step 0]$$$$$$$$$$ Done loading model')

    data_test = []
    data_name = []  # read ids
    batch_idx = 0
    it = 0
    for file in glob.glob(inpath + '/*.fast5'):
        data_test, data_name = get_raw_data(file, data_test, data_name, cutoff, max_seq_len)
        it += 1

        if it == batch:
            print(f'[Step 1]$$$$$$$$$$ Done loading data with batch {str(batch_idx)}, '
                  f'Getting {str(len(data_test))} of sequences')
            data_test = normalize(data_test, batch_idx, max_seq_len)
            process(data_test, data_name, batch_idx, bmodel, outpath, device)
            print(f'[Step 4]$$$$$$$$$$ Done with batch {str(batch_idx)}\n')

            del data_test
            data_test = []
            del data_name
            data_name = []

            batch_idx += 1
            it = 0

    print(f'[Step FINAL] --- {time.time() - start_time} seconds ---')


if __name__ == '__main__':
    main()
