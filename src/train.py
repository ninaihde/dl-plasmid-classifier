import click
import torch
import os

from dataset import Dataset
from model import Bottleneck, ResNet
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def validate(validation_generator, device, model, criterion):
    total_loss, total_acc = 0, 0

    # set gradient calculation off
    with torch.set_grad_enabled(False):
        for val_data, val_labels in validation_generator:
            val_data, val_labels = val_data.to(device), val_labels.to(torch.long).to(device)
            outputs_val = model(val_data)

            loss_val = criterion(outputs_val, val_labels)
            total_loss += loss_val.item()

            acc_val = 100.0 * (val_labels == outputs_val.max(dim=1).indices).float().mean().item()
            total_acc += acc_val

    return total_loss / len(validation_generator), \
           total_acc / len(validation_generator)


def update_stopping_criterion(current_loss, last_loss, trigger_times, patience):
    if current_loss > last_loss:
        trigger_times += 1
    else:
        trigger_times = 0

    print(f'Trigger times: {str(trigger_times)}')
    return trigger_times


@click.command()
@click.option('--p_train', '-pt', help='path of plasmid training set', type=click.Path(exists=True))
@click.option('--p_val', '-pv', help='path of plasmid validation set', type=click.Path(exists=True))
@click.option('--chr_train', '-ct', help='path of chromosome training set', type=click.Path(exists=True))
@click.option('--chr_val', '-cv', help='path of chromosome validation set', type=click.Path(exists=True))
@click.option('--out_folder', '-o', help='output folder in which models and logs are saved', type=click.Path())
@click.option('--interm', '-i', help='file path for model checkpoint (optional)', type=click.Path(exists=True),
              required=False)
@click.option('--patience', '-p', default=2, help='patience (i.e., number of epochs) to wait before early stopping')
@click.option('--batch', '-b', default=1000, help='batch size, default 1000 reads')
@click.option('--n_workers', '-w', default=8, help='number of workers, default 8')
@click.option('--n_epochs', '-e', default=5, help='number of epochs, default 5')
@click.option('--learning_rate', '-l', default=1e-3, help='learning rate, default 1e-3')
def main(p_train, p_val, chr_train, chr_val, out_folder, interm, patience, batch, n_workers, n_epochs, learning_rate):
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(f'{out_folder}/models'):
        os.makedirs(f'{out_folder}/models')
    if not os.path.exists(f'{out_folder}/logs'):
        os.makedirs(f'{out_folder}/logs')

    # TODO: remove tensorboard's logger
    logger = SummaryWriter(log_dir=f'{out_folder}/logs')

    # set parameters
    params = {'batch_size': batch,
              'shuffle': True,
              'num_workers': n_workers}

    # load files
    training_set = Dataset(p_train, chr_train)
    training_generator = DataLoader(training_set, **params)

    validation_set = Dataset(p_val, chr_val)
    validation_generator = DataLoader(validation_set, **params)

    print(f'Number of batches: {str(len(training_generator))}')

    # create new or load pre-trained model
    model = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device)
    if interm is not None:
        model.load_state_dict(torch.load(interm))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    best_model_epoch = 0

    # setup early stopping
    last_loss = 1.0
    trigger_times = 0

    for epoch in range(n_epochs):
        print(f'\nEpoch: {str(epoch)}')

        for i, (train_data, train_labels) in enumerate(training_generator):
            train_data, train_labels = train_data.to(device), train_labels.to(torch.long).to(device)

            # perform forward propagation
            outputs_train = model(train_data)
            loss_train = criterion(outputs_train, train_labels)
            acc_train = 100.0 * (train_labels == outputs_train.max(dim=1).indices).float().mean().item()
            print(f'Batch: {str(i)}, training loss: {str(loss_train.item())}, training accuracy: {str(acc_train)}')
            logger.add_scalar('train-loss', loss_train, i)
            logger.add_scalar('train-acc', acc_train, i)

            # perform backward propagation
            # -> set gradients to zero (to avoid using combination of old and new gradient as new gradient)
            optimizer.zero_grad()
            # -> compute gradients of loss w.r.t. model parameters
            loss_train.backward()
            # -> update parameters of optimizer
            optimizer.step()

        # validate
        current_loss, current_acc = validate(validation_generator, device, model, criterion)

        # save each model
        torch.save(model.state_dict(), f'{out_folder}/models/model_epoch{epoch}.pt')
        print(f'Validation loss: {str(current_loss)}, validation accuracy: {str(current_acc)}')
        logger.add_scalar('val-loss', current_loss, epoch)
        logger.add_scalar('val-acc', current_acc, epoch)

        # update best model
        if best_acc < current_acc:
            best_acc = current_acc
            best_model_epoch = epoch

        # avoid overfitting with early stopping
        trigger_times = update_stopping_criterion(current_loss, last_loss, trigger_times, patience)
        last_loss = current_loss

        if trigger_times >= patience:
            print(f'Training would be early stopped!\n'
                  f'Best model would be reached after {str(best_model_epoch)} epochs')
            # return  # TODO: comment in again if early stopping criterion is optimized & flush and close logger

        # write all pending events to disk
        logger.flush()

    logger.close()
    print(f'Best model reached after epoch no. {str(best_model_epoch)}')


if __name__ == '__main__':
    main()
