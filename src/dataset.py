import torch

from torch.utils.data.dataset import Dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, p_data, chr_data):
        p = torch.load(p_data)
        c = torch.load(chr_data)
        self.data = torch.cat((p, c))
        self.label = torch.cat((torch.zeros(p.shape[0]), torch.ones(c.shape[0])))  # plasmids: 1, chromosomes: 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]
        return X, y
