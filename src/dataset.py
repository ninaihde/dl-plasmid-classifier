"""
This dataset is an extended version of the dataset used in the SquiggleNet project, see
https://github.com/welch-lab/SquiggleNet/blob/master/dataset.py
"""

import torch

from torch.utils.data.dataset import Dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, pos_ids, current_pos_idx, neg_ids, current_neg_idx):
        p = torch.load(filenames[0])
        n = torch.load(filenames[1])
        self.data = torch.cat((p, n))

        # extract number of reads per class (for labels and ID list iteration)
        self.n_pos_reads = p.shape[0]
        self.n_neg_reads = n.shape[0]

        # assign labels (plasmids: 0, chromosomes: 1)
        self.labels = torch.cat((torch.zeros(self.n_pos_reads), torch.ones(self.n_neg_reads)))

        # store read IDs for evaluation of validation data
        # if no read ID file is parsed, no evaluation data is stored during training
        if pos_ids is not None and neg_ids is not None:
            self.ids = pos_ids[current_pos_idx: (current_pos_idx + self.n_pos_reads)] \
                       + neg_ids[current_neg_idx: (current_neg_idx + self.n_neg_reads)]
        else:
            self.ids = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        if self.ids is not None:
            read_id = self.ids[index]
            return X, y, read_id
        else:
            return X, y, None

    def get_n_pos_reads(self):
        return self.n_pos_reads

    def get_n_neg_reads(self):
        return self.n_neg_reads
