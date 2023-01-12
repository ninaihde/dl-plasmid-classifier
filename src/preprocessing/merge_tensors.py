"""
This script merges the .pt files created in prepare_training.py if several batches were used. This is needed for all
training and validation datasets to be able to load data correctly during training.
"""

import click
import glob
import time
import torch

from tqdm import tqdm


@click.command()
@click.option('--data_dir', '-d', help='folder with training or validation tensor files to merge', required=True,
              type=click.Path(exists=True))
def main(data_dir):
    start_time = time.time()
    print(f'Merging tensor files in {data_dir}...')
    merged_tensors = torch.Tensor()
    for tensor_file in tqdm(sorted(glob.glob(f'{data_dir}/*.pt'))):
        current_tensor = torch.load(tensor_file)
        merged_tensors = torch.cat((merged_tensors, current_tensor))

    torch.save(merged_tensors, f'{data_dir}/tensors_merged.pt')
    print(f'Finished merging. Runtime passed: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
