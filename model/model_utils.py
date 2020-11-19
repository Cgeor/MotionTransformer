import torch
import torch.nn as nn
import numpy as np
from model.Transformer import *


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step == 0:
            return 0

        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def to_subsequence_format(src, subseq_n, subseq_l):
    """[bs, input_n, dim] -> [bs, subseq_n, subseq_l, dim]"""
    idx = np.expand_dims(np.arange(subseq_l), axis=0) + np.expand_dims(np.arange(subseq_n), axis=1)
    src = src[:, idx]
    return src

def subsequence_spectral_form(subsequence_format, dct_m):
    """Returns DCT of subsequences. [bs, subseq_n, subseq_l, dim] -> [bs, subseq_n, subseq_l, dim]"""
    bs, subseq_n, subseq_l, _ = subsequence_format.shape
    values = subsequence_format.clone().reshape([bs * subseq_n, subseq_l, -1]) # reshape for matrix multiplication
    values = torch.matmul(dct_m.unsqueeze(dim=0), values) # [bs*subseq_n, subseq_l, dim]
    values = values.reshape([bs, subseq_n, subseq_l, -1])
    return values

def generate_sequence_values(values, dct_m):
    """[bs, dct_n, dim] -> [bs, dct_n, dim]"""
    values = torch.matmul(dct_m.unsqueeze(dim=0), values)
    return values

def src_reformat(src, N, substract_last_pose=False):
    """ [bs, input_n, in_features] -> [bs, in_feature, input_n]"""
    src = src[:, :N]
    src = src.clone()
    last_pose = src[:, -1]
    if substract_last_pose:
        src = src - last_pose.unsqueeze(1)
    return src, last_pose


class Dense_Block1D(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block1D, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_channels=in_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))

        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class ResConvBlock(nn.Module):
    def __init__(self, d_att_model, kernel_size = 11):
        super(ResConvBlock, self).__init__()

        self.convs = clones(nn.Conv1d(in_channels=d_att_model, out_channels=d_att_model, kernel_size=kernel_size, bias=False), 3)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.kernel_size = kernel_size


    def forward(self, x):
        N = x.shape[-1]
        x1 = self.convs[0](x) + x[:, :, self.kernel_size - 1:N]
        x1 = self.leaky_relu(x1)
        x2 = self.convs[1](x1) + x[:, :, 2 * (self.kernel_size - 1):N]
        x2 = self.leaky_relu(x2)
        x3 = self.convs[2](x2) + x[:, :, 3 * (self.kernel_size - 1):N]
        x3 = self.leaky_relu(x3)

        return x3


def generate_velocities(input):
    """[bs, in, D] -> [bs, in D], instant velocity for each timestep"""

    out = torch.roll(input, 1, 1)
    out = out - input

    return out


def sum_vel(input):
    """cumulative sum of velocities"""
    return torch.cumsum(input, 1)

