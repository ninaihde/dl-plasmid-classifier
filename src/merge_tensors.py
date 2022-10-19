import click
import glob
import os
import torch


@click.command()
@click.option('--path', '-p', help='path to preprocessed tensors', type=click.Path(exists=True))
def main(path):
    for ds in [x.split(os.path.sep)[-1] for x in glob.glob(f'{path}/*')
               if x.split(os.path.sep)[-1].startswith('prepared') and not x.split(os.path.sep)[-1].endswith('test')]:
        merged_tensors = torch.Tensor()
        for tensor_file in glob.glob(f'{path}/{ds}/*.pt'):
            current_tensor = torch.load(tensor_file)
            merged_tensors = torch.cat((merged_tensors, current_tensor))

        torch.save(merged_tensors, f'{path}/{ds}/{ds}_merged.pt')
        print(f'Finished {ds}.')


if __name__ == '__main__':
    main()
