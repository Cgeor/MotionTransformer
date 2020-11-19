from torch.nn import Module
from torch import nn
import torch
import math
from model import GCN
import utils.util as util
import numpy as np
import torch
import torch.nn as nn
from model.Transformer import *
from model.model_utils import *


class AttModelRef4(Module):
    def __init__(self, in_features=66, kernel_size=10, d_model=512, device='cuda', h=4, dropout=0.1, dct_n=10, num_stage=12, d_ff = 256, N=6, d_att_model=128):
        super(AttModelRef4, self).__init__()

        self.kernel_size = kernel_size
        self.dct_n = dct_n
        self.h = h
        self.device = device

        c = copy.deepcopy
        self.pos_encoding = PositionalEncoding(d_model=d_att_model, dropout=dropout)
        attn = MultiHeadedAttention(h=h, d_model=d_att_model)
        ff = PositionwiseFeedForward(d_model=d_att_model, d_ff = d_ff, dropout=dropout)
        self.encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.init_lin = nn.Linear(in_features, d_att_model)
        self.inter_lin = nn.Linear(d_att_model, in_features)
        self.inter_dropout = nn.Dropout(dropout)
        self.inter_conv = nn.Sequential(nn.Conv1d(in_channels=d_att_model, out_channels=in_features, kernel_size=11,
                                             bias=False),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=11,
                                             bias=False),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=11,
                                            bias=False))

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param itera:
        :return:
        """
        subseq_nbr = input_n - output_n - self.kernel_size + 1
        subseq_length = self.kernel_size + output_n

        dct_m, idct_m = util.get_dct_matrix(subseq_length)
        dct_m = torch.from_numpy(dct_m).float().to(device=self.device)
        idct_m = torch.from_numpy(idct_m).float().to(device=self.device)

        outputs = []
        batch_size = src.shape[0]
        src, last_pose = src_reformat(src, input_n, substract_last_pose=False) # [32, 50, 66]
        new_in = src.clone()
        att_in = src.clone()
        att_in = self.init_lin(att_in)
        #att_in = self.pos_encoding(att_in)
        att_out = self.encoder(att_in, mask=None) # in : [32, 50, d_model] out : [32, 50, d_model]
        att_out = self.inter_conv(att_out.transpose(1, 2)).transpose(1, 2)
        #att_out = self.inter_dropout(att_out)
        #att_out = att_out + last_pose.unsqueeze(1)
        #att_out = self.inter_lin(att_out)
        att_out = att_out[:, -self.kernel_size - output_n:]
        att_out = att_out + new_in[:, -self.kernel_size - output_n:]
        att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

        idx = list(range(-self.kernel_size, 0, 1)) + [
            -1] * output_n  # [-10, -9, ..., -1, -1 ..., -1], indexes used for GCN input
        input_gcn = src[:, idx]
        dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

        outputs.append(out_gcn.unsqueeze(2))


        if itera > 1:
            for i in range(itera - 1):
                new_in = torch.cat([new_in[:, -input_n + output_n:], out_gcn[:, -output_n:]], dim=1)
                att_in = self.init_lin(new_in)
                att_in = self.pos_encoding(att_in)
                att_out = self.encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
                att_out = self.inter_conv(att_out.transpose(1, 2)).transpose(1, 2)
                #att_out = att_out + last_pose.unsqueeze(1)
                att_out = att_out + new_in[:, -self.kernel_size - output_n:]
                att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)
                input_gcn = new_in[:, idx]
                dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
                dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

                outputs.append(out_gcn[:, -output_n:].unsqueeze(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class AttModelMod(Module):
    def __init__(self, in_features=66, kernel_size=10, d_model=512, device='cuda', h=4, dropout=0.1, dct_n=10, num_stage=12, d_ff = 256, N=8, d_att_model=128):
        super(AttModelMod, self).__init__()

        self.kernel_size = kernel_size
        self.dct_n = dct_n
        self.h = h
        self.device = device

        c = copy.deepcopy
        self.pos_encoding = PositionalEncoding(d_model=in_features, dropout=dropout)
        attn = MultiHeadedAttention(h=h, d_model=d_att_model)
        ff = PositionwiseFeedForward(d_model=d_att_model, d_ff = d_ff, dropout=dropout)
        self.encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.init_lin = nn.Linear(in_features, d_att_model)
        #self.inter_lin = nn.Linear(d_att_model, in_features)
        self.inter_conv = nn.Sequential(nn.Conv1d(in_channels=d_att_model, out_channels=in_features, kernel_size=11,
                                             bias=False),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=11,
                                             bias=False),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=11,
                                            bias=False),
                                   nn.LeakyReLU(0.1))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param itera:
        :return:
        """
        subseq_nbr = input_n - output_n - self.kernel_size + 1
        subseq_length = self.kernel_size + output_n

        dct_m, idct_m = util.get_dct_matrix(subseq_length)
        dct_m = torch.from_numpy(dct_m).float().to(device=self.device)
        idct_m = torch.from_numpy(idct_m).float().to(device=self.device)

        outputs = []
        batch_size = src.shape[0]
        src, last_pose = src_reformat(src, input_n, substract_last_pose=False) # [32, 50, 66]
        new_in = src.clone()
        att_in = src.clone()
        att_in = self.pos_encoding(att_in)
        att_in = self.init_lin(att_in)
        att_out = self.encoder(att_in, mask=None) # in : [32, 50, d_model] out : [32, 50, d_model]
        att_out = self.inter_conv(att_out.transpose(1, 2)).transpose(1, 2)
        #att_out = self.inter_lin(att_out)
        #att_out = att_out[:, -self.kernel_size - output_n:]
        att_out = att_out + new_in[:, -self.kernel_size - output_n:]
        att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

        idx = list(range(-self.kernel_size, 0, 1)) + [
            -1] * output_n  # [-10, -9, ..., -1, -1 ..., -1], indexes used for GCN input
        input_gcn = src[:, idx]
        dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

        outputs.append(out_gcn.unsqueeze(2))

        if itera > 1:
            for i in range(itera - 1):
                new_in = torch.cat([new_in[:, -input_n + output_n:], out_gcn[:, -output_n:]], dim=1)
                att_in = self.init_lin(new_in)
                att_out = self.encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
                att_out = self.inter_conv(att_out.transpose(1, 2)).transpose(1, 2)
                att_out = att_out + new_in[:, -self.kernel_size - output_n:]
                att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)
                input_gcn = new_in[:, idx]
                dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
                dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

                outputs.append(out_gcn[:, -output_n:].unsqueeze(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class AttModelModVel(Module):
    def __init__(self, in_features=66, kernel_size=10, d_model=512, device='cuda', h=4, dropout=0.1, dct_n=10, num_stage=12, d_ff = 256, N=6, d_att_model=128):
        super(AttModelModVel, self).__init__()

        self.kernel_size = kernel_size
        self.dct_n = dct_n
        self.h = h
        self.device = device
        self.in_features = in_features

        c = copy.deepcopy
        self.pos_encoding = PositionalEncoding(d_model=in_features, dropout=dropout)
        attn = MultiHeadedAttention(h=h, d_model=d_att_model)
        ff = PositionwiseFeedForward(d_model=d_att_model, d_ff = d_ff, dropout=dropout)
        self.encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.init_lin = nn.Linear(in_features, d_att_model)
        #self.inter_lin = nn.Linear(d_att_model, in_features)
        self.inter_conv = nn.Sequential(nn.Conv1d(in_channels=d_att_model, out_channels=in_features, kernel_size=11,
                                             bias=False),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=11,
                                             bias=False),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=11,
                                            bias=False),
                                   nn.LeakyReLU(0.1))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param itera:
        :return:
        """
        subseq_nbr = input_n - output_n - self.kernel_size + 1
        subseq_length = self.kernel_size + output_n

        dct_m, idct_m = util.get_dct_matrix(subseq_length)
        dct_m = torch.from_numpy(dct_m).float().to(device=self.device)
        idct_m = torch.from_numpy(idct_m).float().to(device=self.device)

        outputs = []
        batch_size = src.shape[0]
        src, last_pose = src_reformat(src, input_n, substract_last_pose=False) # [32, 50, 66]
        dt = torch.from_numpy(2.0 - np.exp(-np.arange(10))).to(device=self.device).expand(self.in_features, output_n).transpose(0, 1)
        vel = src[:, -1] - src[:, -2]
        dx = vel.unsqueeze(1) * dt

        new_in = src.clone()
        att_in = src.clone()
        att_in = self.pos_encoding(att_in)
        att_in = self.init_lin(att_in)
        att_out = self.encoder(att_in, mask=None) # in : [32, 50, d_model] out : [32, 50, d_model]
        att_out = self.inter_conv(att_out.transpose(1, 2)).transpose(1, 2)
        #att_out = self.inter_lin(att_out)
        #att_out = att_out[:, -self.kernel_size - output_n:]
        att_out = att_out + new_in[:, -self.kernel_size - output_n:]
        att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

        idx = list(range(-self.kernel_size, 0, 1)) + [
            -1] * output_n  # [-10, -9, ..., -1, -1 ..., -1], indexes used for GCN input
        input_gcn = src[:, idx]
        input_gcn[:, self.kernel_size:] = input_gcn[:, self.kernel_size:] + dx  # add velocity
        dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0), dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

        outputs.append(out_gcn.unsqueeze(2))

        if itera > 1:
            for i in range(itera - 1):
                new_in = torch.cat([new_in[:, -input_n + output_n:], out_gcn[:, -output_n:]], dim=1)
                att_in = self.init_lin(new_in)
                att_out = self.encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
                att_out = self.inter_conv(att_out.transpose(1, 2)).transpose(1, 2)
                att_out = att_out + new_in[:, -self.kernel_size - output_n:]
                att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)
                input_gcn = new_in[:, idx]
                dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
                dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

                outputs.append(out_gcn[:, -output_n:].unsqueeze(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class AttModelResNet(Module):
    def __init__(self, in_features=66, kernel_size=10, d_model=512, device='cuda', h=4, dropout=0.1, dct_n=10,
                 num_stage=12, d_ff=128, N=4, d_att_model=128):
        super(AttModelResNet, self).__init__()

        self.kernel_size = kernel_size
        self.dct_n = dct_n
        self.h = h
        self.device = device

        c = copy.deepcopy
        self.pos_encoding = PositionalEncoding(d_model=d_att_model, dropout=dropout)
        attn = MultiHeadedAttention(h=h, d_model=d_att_model)
        ff = PositionwiseFeedForward(d_model=d_att_model, d_ff=d_ff, dropout=dropout)
        self.encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.init_lin = nn.Linear(in_features, d_att_model)
        self.inter_lin = nn.Linear(d_att_model, in_features)
        self.res_conv_block = ResConvBlock(in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param itera:
        :return:
        """
        subseq_nbr = input_n - output_n - self.kernel_size + 1
        subseq_length = self.kernel_size + output_n

        dct_m, idct_m = util.get_dct_matrix(subseq_length)
        dct_m = torch.from_numpy(dct_m).float().to(device=self.device)
        idct_m = torch.from_numpy(idct_m).float().to(device=self.device)

        outputs = []
        batch_size = src.shape[0]
        src, last_pose = src_reformat(src, input_n, substract_last_pose=False)  # [32, 50, 66]

        new_in = src.clone()
        att_in = src.clone()
        att_in = self.init_lin(att_in)
        att_in = self.pos_encoding(att_in)
        att_out = self.encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
        att_out = self.inter_lin(att_out)
        att_out = self.res_conv_block(att_out.transpose(1, 2)).transpose(1, 2)

        idx = list(range(-self.kernel_size, 0, 1)) + [
            -1] * output_n  # [-10, -9, ..., -1, -1 ..., -1], indexes used for GCN input
        input_gcn = src[:, idx]

        att_out = att_out + input_gcn
        att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

        dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

        outputs.append(out_gcn.unsqueeze(2))

        if itera > 1:
            for i in range(itera - 1):
                new_in = torch.cat([new_in[:, -input_n + output_n:], out_gcn[:, -output_n:]], dim=1)
                att_in = self.init_lin(new_in)
                att_in = self.pos_encoding(att_in)
                att_out = self.encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
                att_out = self.inter_lin(att_out)
                att_out = self.res_conv_block(att_out.transpose(1, 2)).transpose(1, 2)
                input_gcn = new_in[:, idx]
                att_out = att_out + input_gcn
                att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

                dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
                dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

                outputs.append(out_gcn[:, -output_n:].unsqueeze(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs



class AttModelResNetVel(Module):
    def __init__(self, in_features=66, kernel_size=10, d_model=512, device='cuda', h=4, dropout=0.1, dct_n=10,
                 num_stage=12, d_ff=128, N=4, d_att_model=128):
        super(AttModelResNetVel, self).__init__()

        self.kernel_size = kernel_size
        self.dct_n = dct_n
        self.h = h
        self.device = device

        c = copy.deepcopy
        self.pos_encoding = PositionalEncoding(d_model=d_att_model, dropout=dropout)
        attn = MultiHeadedAttention(h=h, d_model=d_att_model)
        ff = PositionwiseFeedForward(d_model=d_att_model, d_ff=d_ff, dropout=dropout)
        self.pos_encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)
        self.vel_encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.init_lin = nn.Linear(in_features, d_att_model)
        self.inter_lin = nn.Linear(d_att_model, in_features)
        self.res_conv_block = ResConvBlock(in_features)

        self.init_lin_vel = nn.Linear(in_features, d_att_model)
        self.inter_lin_vel = nn.Linear(d_att_model, in_features)
        self.res_conv_block_vel = ResConvBlock(in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param itera:
        :return:
        """
        subseq_nbr = input_n - output_n - self.kernel_size + 1
        subseq_length = self.kernel_size + output_n

        dct_m, idct_m = util.get_dct_matrix(subseq_length)
        dct_m = torch.from_numpy(dct_m).float().to(device=self.device)
        idct_m = torch.from_numpy(idct_m).float().to(device=self.device)

        outputs = []
        batch_size = src.shape[0]
        src, last_pose = src_reformat(src, input_n, substract_last_pose=False)  # [32, 50, 66]
        vel = src - torch.roll(src, 1, dims=1)
        vel[:, 0] = vel[:, 0] / input_n

        new_in = src.clone()
        att_in = src.clone()
        att_in = self.init_lin(att_in)
        att_in = self.pos_encoding(att_in)
        att_out = self.pos_encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
        att_out = self.inter_lin(att_out)
        att_out = self.res_conv_block(att_out.transpose(1, 2)).transpose(1, 2)

        att_in_vel = self.init_lin_vel(vel)
        att_in_vel = self.pos_encoding(att_in_vel)
        att_out_vel = self.vel_encoder(att_in_vel, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
        att_out_vel = self.inter_lin_vel(att_out_vel)
        att_out_vel = self.res_conv_block_vel(att_out_vel.transpose(1, 2)).transpose(1, 2)

        idx = list(range(-self.kernel_size, 0, 1)) + [
            -1] * output_n  # [-10, -9, ..., -1, -1 ..., -1], indexes used for GCN input
        input_gcn = src[:, idx]

        att_out = att_out + input_gcn
        input_gcn = input_gcn + att_out_vel
        att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)
        dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

        dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

        outputs.append(out_gcn.unsqueeze(2))

        if itera > 1:
            for i in range(itera - 1):
                new_in = torch.cat([new_in[:, -input_n + output_n:], out_gcn[:, -output_n:]], dim=1)
                vel = new_in - torch.roll(new_in, 1, dims=1)
                vel[:, 0] = vel[:, 0] / input_n
                att_in = self.init_lin(new_in)
                att_in = self.pos_encoding(att_in)
                att_out = self.pos_encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
                att_out = self.inter_lin(att_out)
                att_out = self.res_conv_block(att_out.transpose(1, 2)).transpose(1, 2)

                att_in_vel = self.init_lin_vel(vel)
                att_in_vel = self.pos_encoding(att_in_vel)
                att_out_vel = self.vel_encoder(att_in_vel, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]
                att_out_vel = self.inter_lin_vel(att_out_vel)
                att_out_vel = self.res_conv_block_vel(att_out_vel.transpose(1, 2)).transpose(1, 2)
                input_gcn = new_in[:, idx]

                att_out = att_out + input_gcn
                input_gcn = input_gcn + att_out_vel
                att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)
                dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

                dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

                outputs.append(out_gcn[:, -output_n:].unsqueeze(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class AttModelResNet2ppl(Module):
    def __init__(self, in_features=66, kernel_size=10, d_model=512, device='cuda', h=2, dropout=0.1, dct_n=10,
                 num_stage=12, d_ff=128, N=4, d_att_model=128):
        super(AttModelResNet2ppl, self).__init__()

        self.kernel_size = kernel_size
        self.dct_n = dct_n
        self.h = h
        self.device = device

        c = copy.deepcopy
        self.pos_encoding = PositionalEncoding(d_model=d_att_model, dropout=dropout)
        attn = MultiHeadedAttention(h=h, d_model=d_att_model)
        ff = PositionwiseFeedForward(d_model=d_att_model, d_ff=d_ff, dropout=dropout)
        self.encoder = Encoder(EncoderLayer(d_att_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_att_model, c(attn), c(attn), c(ff), dropout), N)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.init_lin_encode = nn.Linear(in_features, d_att_model)
        self.init_lin_decode = nn.Linear(in_features, d_att_model)
        self.inter_lin = nn.Linear(d_att_model, in_features)
        self.res_conv_block = ResConvBlock(in_features)

    def forward(self, src, src2, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param itera:
        :return:
        """
        subseq_nbr = input_n - output_n - self.kernel_size + 1
        subseq_length = self.kernel_size + output_n

        dct_m, idct_m = util.get_dct_matrix(subseq_length)
        dct_m = torch.from_numpy(dct_m).float().to(device=self.device)
        idct_m = torch.from_numpy(idct_m).float().to(device=self.device)

        outputs = []
        batch_size = src.shape[0]
        src, last_pose = src_reformat(src, input_n, substract_last_pose=False)  # [32, 50, 66]

        encode_in = src.clone()
        decode_in = src.clone()
        encode_in = self.init_lin_encode(encode_in)
        encode_in = self.pos_encoding(encode_in)
        encode_out = self.encoder(encode_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]

        decode_in = self.init_lin_encode(decode_in)
        decode_in = self.pos_encoding(decode_in)
        decode_out = self.decoder(self, decode_in, encode_out, src_mask = None, tgt_mask = None)

        att_out = self.inter_lin_decode(decode_out)
        att_out = self.res_conv_block(att_out.transpose(1, 2)).transpose(1, 2)

        idx = list(range(-self.kernel_size, 0, 1)) + [
            -1] * output_n  # [-10, -9, ..., -1, -1 ..., -1], indexes used for GCN input
        input_gcn = src[:, idx]

        att_out = att_out + input_gcn
        att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

        dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

        outputs.append(out_gcn.unsqueeze(2))

        if itera > 1:
            for i in range(itera - 1):
                new_in = torch.cat([new_in[:, -input_n + output_n:], out_gcn[:, -output_n:]], dim=1)
                att_in = self.init_lin(new_in)
                att_in = self.pos_encoding(att_in)
                att_out = self.encoder(att_in, mask=None)  # in : [32, 50, d_model] out : [32, 50, d_model]

                att_out = self.inter_lin(att_out)
                att_out = self.res_conv_block(att_out.transpose(1, 2)).transpose(1, 2)
                input_gcn = new_in[:, idx]
                att_out = att_out + input_gcn
                att_out = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), att_out).transpose(1, 2)

                dct_in_tmp = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
                dct_in_tmp = torch.cat([dct_in_tmp, att_out], dim=-1)
                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :self.dct_n].transpose(1, 2))

                outputs.append(out_gcn[:, -output_n:].unsqueeze(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs
