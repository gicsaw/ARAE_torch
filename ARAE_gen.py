#!/usr/bin/env python
from model import ARAE as ARAE
from ZINC.char import char_list, char_dict
import os
import sys
import numpy as np
import math

import torch
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


def vec_to_char(out_num):
    stri = ""
    for cha in out_num:
        stri += char_list[cha]
    return stri


def main():

    Ntest = 10000

    Nfea = len(char_list)

    Nseq = 110
    hidden_dim = 300
    seed_dim = hidden_dim
    NLSTM_layer = 1
    batch_size = 100

    N_batch = int(math.ceil(Ntest/batch_size))
    print(N_batch, Ntest)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_num = torch.cuda.current_device()
        print(device_num)
        device = torch.device("cuda:%d" % device_num)
        torch.set_num_threads(2)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(24)
#    device =  torch.device("cpu")
    print(device)

    para = {'Nseq': Nseq, 'Nfea': Nfea, 'hidden_dim': hidden_dim,
            'seed_dim': seed_dim, 'NLSTM_layer': NLSTM_layer, 'device': device}

    model = ARAE.Net(para)
    model.to(device)

    save_dir = "save_ARAE"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_result_dir = "result_ARAE_gen"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    total_st = time.time()

    std0 = 0.0
    std = 0.0
    std_seed = 0.25
    std_decay_ratio = 0.99
    mean0 = torch.zeros(batch_size, hidden_dim)
    mean_seed = torch.zeros(batch_size, seed_dim)

#    epoch_list=[9]
    epoch_list = [9, 19, 29, 39, 49, 59, 69, 79, 89]

    for epoch in epoch_list:
        path = save_dir+"/save_%d.pth" % epoch
        model = torch.load(path)
        model.to(device)

        st = time.time()
        save_result_dir2 = save_result_dir+"/epoch%d" % epoch
        if not os.path.exists(save_result_dir2):
            os.makedirs(save_result_dir2)

        file_ARAE = save_result_dir2+"/ARAE_smiles.txt"
        fp_ARAE = open(file_ARAE, "w")

        model.eval()
        for i in range(0, N_batch):
            ini = batch_size*i
            fin = batch_size*(i+1)
            if fin > Ntest:
                fin = Ntest
            mm = np.arange(ini, fin)
            b_size = mm.shape[0]

            batch_s = torch.normal(mean=mean_seed, std=std_seed).to(device)
            Z_gen = model.Gen(batch_s)

            out_num_ARAE = model.Dec.decoding(Z_gen)

#            for k in range(0,10):
#                out_string=vec_to_char(out_num_ARAE[k])
#                print("ARAE: ",out_string)

            for k in range(0, b_size):
                line_ARAE = vec_to_char(out_num_ARAE[k])+"\n"
                fp_ARAE.write(line_ARAE)
        fp_ARAE.close()

        et = time.time()
        print("time: %10.2f" % (et-st))

    print('Finished Training')
    total_et = time.time()
    print("time : %10.2f" % (total_et-total_st))


if __name__ == "__main__":
    main()
