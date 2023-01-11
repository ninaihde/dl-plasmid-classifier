"""
This dataset is an extended version of the dataset used in the SquiggleNet project, see
https://github.com/welch-lab/SquiggleNet/blob/master/dataset.py
"""

import glob
import torch

from torch.utils.data.dataset import Dataset


class CustomizedDataset(torch.utils.data.Dataset):
    def __init__(self, p_data, chr_data, p_ids=None, chr_ids=None):
        p_files = sorted([f for f in glob.glob(f'{p_data}/*.pt') if not f.endswith('tensors_merged.pt')])
        c_files = sorted([f for f in glob.glob(f'{chr_data}/*.pt') if not f.endswith('tensors_merged.pt')])
        self.current_file = ''
        self.reads = torch.Tensor()

        # create dictionary storing which indices given into __getitem__ (i.e., which reads) are in which files
        overall_read_count = 0
        self.idx_ranges_per_file = dict()
        for file in p_files + c_files:
            read_count = torch.load(file).shape[0]
            overall_read_count += read_count
            self.idx_ranges_per_file[overall_read_count - read_count, overall_read_count - 1] = file

            # extract number of reads per class
            if file == p_files[-1]:
                p_count = overall_read_count
        c_count = overall_read_count - p_count

        # extract class counts for loss balancing and label creation
        self.class_counts = p_count, c_count
        self.labels = torch.cat((torch.zeros(p_count), torch.ones(c_count)))  # plasmids: 0, chromosomes: 1

        # store read IDs for evaluation of validation data
        # if no read ID file is parsed, no evaluation is performed
        if p_ids is not None and chr_ids is not None:
            self.ids = open(p_ids, 'r').read().split('\n')[:-1] + open(chr_ids, 'r').read().split('\n')[:-1]
        else:
            self.ids = list()

    def __len__(self):
        return len(self.labels)

    def get_file(self, index):
        for start, end in self.idx_ranges_per_file.keys():
            if start <= index <= end:
                return self.idx_ranges_per_file[start, end], start

    def get_read_position(self, index, start):
        if start != 0:
            return index - start
        else:
            return index

    def __getitem__(self, index):
        file, start = self.get_file(index)
        read_pos = self.get_read_position(index, start)

        if file != self.current_file:
            self.reads = torch.load(file)
            self.current_file = file

        X = self.reads[read_pos]
        y = self.labels[index]

        if self.ids:
            read_id = self.ids[index]
            return X, y, read_id
        else:
            return X, y, 'undefined'

    def get_class_counts(self):
        return self.class_counts
