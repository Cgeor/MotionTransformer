#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
#from progress.bar import Bar
import pandas as pd
from matplotlib import pyplot as plt

#from utils import loss_funcs, utils as utils
from utils.opt import Options
import utils.data_utils as data_utils
import utils.viz as viz

from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim



def main(opt):
    is_cuda = torch.cuda.is_available()

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    itera = 50

    #model = nnmodel.GCN(input_feature=(input_n + output_n), hidden_feature=opt.linear_size, p_dropout=opt.dropout, num_stage=opt.num_stage, node_n=48)
    model = AttModel.AttModelRef4(in_features=opt.in_features, kernel_size=opt.kernel_size, d_model=opt.d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n, device=opt.device)
    if is_cuda:
        model.cuda()
    model_path_len = '/home/costa/src/Transformer/checkpoint/trans_N6_last_pose_in50_out10_ks10_dctn20/ckpt_best.pth.tar'
    model_path_len = '/home/costa/Desktop/ckpt_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    if is_cuda:
        ckpt = torch.load(model_path_len)
    else:
        ckpt = torch.load(model_path_len, map_location='cpu')
    err_best = ckpt['err']
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    acts = data_utils.define_actions('all')
    test_data = dict()

    for act in acts:
        test_dataset = datasets.Datasets(opt=opt, actions=[act], split=1, itera=itera)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            pin_memory=True)

    dim_used = test_dataset.dimensions_to_use
    print(">>> data loaded !")

    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for act in acts:
        for i, all_seq in enumerate(test_data[act]):
            inputs = Variable(all_seq[:, :opt.input_n, dim_used]).float()
            all_seq = Variable(all_seq).float()
            if is_cuda:
                inputs = inputs.cuda()
                all_seq = all_seq.cuda()

            outputs = model(inputs, opt.output_n, opt.input_n, itera=itera)

            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)
            '''
            _, idct_m = data_utils.get_dct_matrix(seq_len)
            if is_cuda:
                idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
            else:
                idct_m = Variable(torch.from_numpy(idct_m)).float()

            outputs_t = outputs.view(-1, seq_len).transpose(0, 1)
            outputs_exp = torch.matmul(idct_m, outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len, seq_len).transpose(1, 2)

            p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
            p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]

            pred_expmap = all_seq.clone()
            dim_used = np.array(dim_used)
            pred_expmap[:, :, dim_used] = outputs_exp
            '''
            pred = all_seq.clone()
            pred[:, opt.input_n:opt.input_n + opt.output_n*itera, dim_used] = outputs[:, opt.kernel_size:, 0]
            pred[:, opt.input_n:opt.input_n + opt.output_n*itera, index_to_ignore] = pred[:, opt.input_n:opt.input_n + opt.output_n*itera, index_to_equal]
            targ = all_seq
            pred = pred.cpu().data.numpy()
            targ = targ.cpu().data.numpy()
            for k in range(8):
                plt.cla()
                figure_title = "action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions_from_3d(targ[k, opt.input_n:opt.input_n + opt.output_n*itera, :], pred[k, opt.input_n:opt.input_n + opt.output_n*itera, :], fig, ax, figure_title)
                plt.pause(1)


if __name__ == "__main__":
    option = Options().parse()
    main(option)