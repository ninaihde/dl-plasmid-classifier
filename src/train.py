import click
import torch
import os

from dataset import Dataset
from model import Bottleneck, ResNet
from torch import nn
from torch.utils.data import DataLoader


@click.command()
@click.option('--p_train', '-tt', help='path of plasmid training set', type=click.Path(exists=True))
@click.option('--p_val', '-tv', help='path of plasmid validation set', type=click.Path(exists=True))
@click.option('--chr_train', '-nt', help='path of chromosome training set', type=click.Path(exists=True))
@click.option('--chr_val', '-nv', help='path of chromosome validation set', type=click.Path(exists=True))
@click.option('--outpath', '-op', help='output path for the best trained model', type=click.Path())
@click.option('--outname', '-on', help='filename of the best trained model (specify without filename extension)')
@click.option('--interm', '-i', help='file path for model checkpoint (optional)', type=click.Path(exists=True),
              required=False)
@click.option('--batch', '-b', default=1000, help='batch size, default 1000 reads')
@click.option('--n_worker', '-w', default=4, help='number of workers, default 4')
@click.option('--epoch', '-e', default=20, help='number of epoches, default 20')
@click.option('--learning_rate', '-l', default=1e-3, help='learning rate, default 1e-3')
def main(p_train, p_val, chr_train, chr_val, outpath, outname, interm, batch, n_worker, epoch, learning_rate):
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # set parameters
    params = {'batch_size': batch,
              'shuffle': True,
              'num_workers': n_worker}

    # load files
    training_set = Dataset(p_train, chr_train)
    training_generator = DataLoader(training_set, **params)

    validation_set = Dataset(p_val, chr_val)
    validation_generator = DataLoader(validation_set, **params)

    # create or load pre-trained model
    layers = [2, 2, 2, 2]
    model = ResNet(Bottleneck, layers).to(device)
    if interm is not None:
        model.load_state_dict(torch.load(interm))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    best_model = None
    i = 0

    # iterate epochs of training
    for epoch in range(epoch):
        for train_data, train_labels in training_generator:
            train_data, train_labels = train_data.to(device), train_labels.to(torch.long).to(device)

            ### forward pass
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            acc_train = 100.0 * (train_labels == outputs.max(dim=1).indices).float().mean().item()
            print(f'training accuracy: {acc_train}')

            ### validation (set gradient calculation off)
            with torch.set_grad_enabled(False):
                avg_acc_val = 0
                val_i = 0
                for val_data, val_labels in validation_generator:
                    val_data, val_labels = val_data.to(device), val_labels.to(device)
                    outputs_val = model(val_data)
                    acc_val = 100.0 * (val_labels == outputs_val.max(dim=1).indices).float().mean().item()
                    val_i += 1
                    avg_acc_val += acc_val
                avg_acc_val = avg_acc_val / val_i

                if best_acc < avg_acc_val:
                    best_acc = avg_acc_val
                    best_model = model
                    torch.save(best_model.state_dict(), f'{outpath}/{outname}.pt')

                print(f'epoch: {str(epoch)}, i: {str(i)}, best validation accuracy: {str(best_acc)}')
                i += 1

            ### backward pass and optimize
            # set gradients to zero (to avoid using combination of old and new gradient as new gradient)
            optimizer.zero_grad()
            # compute gradients of loss w.r.t. model parameters
            loss.backward()
            # update parameters of optimizer
            optimizer.step()


if __name__ == '__main__':
    main()
