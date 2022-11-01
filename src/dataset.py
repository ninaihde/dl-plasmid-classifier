import torch

from torch.utils.data.dataset import Dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, p_data, chr_data, p_ids=None, chr_ids=None):
        p = torch.load(p_data)
        c = torch.load(chr_data)
        self.data = torch.cat((p, c))
        self.class_counts = len(p), len(c)
        self.labels = torch.cat((torch.zeros(p.shape[0]), torch.ones(c.shape[0])))  # plasmids: 0, chromosomes: 1

        if p_ids is not None and chr_ids is not None:
            self.ids = open(p_ids, 'r').read().split('\n')[:-1] + open(chr_ids, 'r').read().split('\n')[:-1]
        else:
            self.ids = list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        if self.ids:
            ID = self.ids[index]
            return X, y, ID
        else:
            return X, y

    def get_class_counts(self):
        return self.class_counts
