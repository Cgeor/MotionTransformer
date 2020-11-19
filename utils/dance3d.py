from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch
import json


class Datasets(Dataset):

    def __init__(self, opt, split=0, itera=1):
        """
        :param path_to_data:
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = "/home/costa/src/DataPreprocessing/out_opt.json"
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n * itera
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n

        with open(self.path_to_data) as file:
            the_sequence = json.load(file)

        seq_p1 = np.array(the_sequence['p1'])
        seq_p2 = np.array(the_sequence['p2'])
        seq_p1 = seq_p1.reshape(seq_p1.shape[0], -1) # (25, 3) to (75)
        seq_p2 = seq_p2.reshape(seq_p2.shape[0], -1)

        n, d = seq_p1.shape
        even_list = range(0, n, self.sample_rate)
        num_frames = len(even_list)
        seq_p1 = seq_p1[even_list, :]
        seq_p2 = seq_p2[even_list, :]
        seq_p1 = torch.from_numpy(seq_p1).float().to(device=opt.device)
        seq_p2 = torch.from_numpy(seq_p2).float().to(device=opt.device)

        self.p3d[0] = seq_p1
        self.p3d[1] = seq_p2

        valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)
        self.data_idx = list(valid_frames)


    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[0][fs], self.p3d[1][fs]
