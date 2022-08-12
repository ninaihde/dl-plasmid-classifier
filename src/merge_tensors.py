import click
import glob
import torch


@click.command()
@click.option('--path', '-p', help='path to preprocessed tensors', type=click.Path(exists=True))
def main(path):
    # TODO: change '\\' to '/'
    for ds in [x.split('\\')[-1] for x in glob.glob(f'{path}/*') if x.split('\\')[-1].startswith('prepared')
                                                                    and not x.split('\\')[-1].endswith('test')]:
        merged_tensors = torch.Tensor()
        for tensor_file in glob.glob(f'{path}/{ds}/*.pt'):
            current_tensor = torch.load(tensor_file)
            merged_tensors = torch.cat((merged_tensors, current_tensor))

        torch.save(merged_tensors, f'{path}/{ds}/{ds}_merged.pt')


if __name__ == '__main__':
    main()
