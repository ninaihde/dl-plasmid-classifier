"""
The CustomDataLoader works like a wrapper of PyTorch's DataLoader class. It creates one DataLoader per file (i.e., loads
the file) at the moment the respective file should be processed. This avoids loading/ storing all data at once and
ensures that each file is loaded only once per epoch.
"""

import glob
import torch

from dataset import CustomDataset
from torch.utils.data import DataLoader


class CustomDataLoader:
    def __init__(self, pos_dir, neg_dir, params, random_gen, pos_ids=None, neg_ids=None):
        # init files
        p_files = glob.glob(f'{pos_dir}/*.pt')
        c_files = glob.glob(f'{neg_dir}/*.pt')
        self.files = list(zip(p_files, c_files))
        self.current_file = None
        self.current_file_idx = 0

        # set read ID lists (if wanted for validation)
        if pos_ids is not None and neg_ids is not None:
            self.pos_ids = open(pos_ids, 'r').read().split('\n')[:-1]
            self.neg_ids = open(neg_ids, 'r').read().split('\n')[:-1]
        else:
            self.pos_ids = None
            self.neg_ids = None

        # init indices to know where we are in read ID lists
        self.current_pos_idx = 0
        self.current_neg_idx = 0

        # extract number of reads per class
        self.class_counts = {'pos': 0, 'neg': 0}
        for pos_file, neg_file in self.files:
            self.class_counts['pos'] += torch.load(pos_file).shape[0]
            self.class_counts['neg'] += torch.load(neg_file).shape[0]

        # set parameters needed for PyTorch's DataLoader
        self.params = params

        # create random generator for file shuffling
        self.random_gen = random_gen

    def __iter__(self):
        self.shuffle_files()
        return self

    def __next__(self):
        # initially setup first file
        if self.current_file is None:
            self.current_file = self.load_next_file()

        # try to get next read in current file
        try:
            return next(self.current_file)
        # if file is completely processed, load next file and extract its first read
        except StopIteration:
            self.current_file = self.load_next_file()
            return next(self.current_file)

    def get_n_reads(self):
        return sum(self.class_counts.values())

    def get_class_counts(self):
        return list(self.class_counts.values())

    def shuffle_files(self):
        self.random_gen.shuffle(self.files)

    def load_next_file(self):
        if len(self.files) <= self.current_file_idx:
            # reset everything for next epoch
            self.current_file = None
            self.current_file_idx = 0
            self.current_pos_idx = 0
            self.current_neg_idx = 0

            raise StopIteration

        dataset = CustomDataset(self.files[self.current_file_idx], self.pos_ids, self.current_pos_idx, self.neg_ids,
                                self.current_neg_idx)
        self.current_pos_idx += dataset.get_n_pos_reads()
        self.current_neg_idx += dataset.get_n_neg_reads()
        self.current_file_idx += 1

        return iter(DataLoader(dataset, **self.params))
