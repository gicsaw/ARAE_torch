#!/usr/bin/env python
from model import CARAE as CARAE
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
from torch.utils.data import Dataset, DataLoader

import time


def accu(pred, val, batch_l):

    correct = 0
    total = 0
    cor_seq = 0
    for i in range(0, batch_l.shape[0]):
        try:
            mm = (pred[i, 0:batch_l[i]].cpu().data.numpy()
                  == val[i, 0:batch_l[i]].cpu().data.numpy())
            correct += mm.sum()
            total += batch_l[i].sum()
            cor_seq += mm.all()
        except:
            print(pred[i, 0:batch_l[i]].cpu().data.numpy(),
                  val[i, 0:batch_l[i]].cpu().data.numpy())
            return 0, 0

    acc = correct/float(total)
    acc2 = cor_seq/batch_l.shape[0]
#    print(correct,total,acc,cor_seq)
    return acc, acc2


def vec_to_char(out_num):
    stri = ""
    for cha in out_num:
        stri += char_list[cha]
    return stri


def cal_prec_rec(Ypred, Ydata, conf):

    small = 0.0000000001
    Ypred0 = Ypred.cpu().data.numpy()
    Ydata0 = Ydata.cpu().data.numpy()
    Ypred00 = Ypred0 > conf
    mm = Ypred00*Ydata0
    TP = mm.sum()
    A = Ydata0.sum()
    P = Ypred00.sum()
    precision = (TP+small)/(P+small)
    recall = (TP+small)/A

    return precision, recall


class UserDataset(Dataset):
    def __init__(self, datadir, dname):

        Xdata_file = datadir+"/X"+dname+".npy"
        self.Xdata = torch.tensor(np.load(Xdata_file), dtype=torch.long)
        Ldata_file = datadir+"/L"+dname+".npy"
        self.Ldata = torch.tensor(np.load(Ldata_file), dtype=torch.long)
        PRdata_file = datadir+"/P"+dname+".npy"
        Pdata_reg0 = np.load(PRdata_file)

        self.Pdata = torch.tensor(np.concatenate(
            [Pdata_reg0[:, 0:1], Pdata_reg0[:, 3:4], Pdata_reg0[:, 2:3]],
            axis=1), dtype=torch.float32)
        self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        return (self.Xdata[index], self.Ldata[index],
                self.Pdata[index])

    def __len__(self):
        return self.len


def main():

    datadir = "data"
    Nfea = len(char_list)
    test_data = UserDataset(datadir, "test")

    Ntest = len(test_data)

    Nseq = 110
    Ncla = 0
    Nreg = 3
    Nprop = Ncla+Nreg

    hidden_dim = 300
    seed_dim = hidden_dim
    NLSTM_layer = 1
    batch_size = 100

    conf = 0.5

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

    para = {'Nseq': Nseq, 'Nfea': Nfea, 'Nprop': Nprop, 'hidden_dim': hidden_dim,
            'seed_dim': seed_dim, 'NLSTM_layer': NLSTM_layer, 'device': device}

    model = CARAE.Net(para)
    model.to(device)

    save_dir = "save_CARAE"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_result_dir = "result_CARAE"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    total_st = time.time()

    criterion_AE = nn.CrossEntropyLoss()
    criterion_Pre_cla = nn.CrossEntropyLoss()
    criterion_Pre_reg = nn.MSELoss()

    AE_parameters = list(model.Enc.parameters())+list(model.Dec.parameters())
    optimizer_AE = optim.Adam(AE_parameters, lr=0.001)
#    optimizer_AE = optim.SGD(AE_parameters, lr=1.0)
    optimizer_gen = optim.Adam(model.Gen.parameters(), lr=0.00001)
    optimizer_cri = optim.Adam(model.Cri.parameters(), lr=0.000002)
    optimizer_Pre = optim.Adam(model.Pre.parameters(), lr=0.001)
    optimizer_Enc_Pre = optim.Adam(model.Enc.parameters(), lr=0.001)
#    optimizer_Pre = optim.SGD(model.Pre.parameters(), lr=1.0)
#    optimizer_Enc_Pre = optim.SGD(model.Enc.parameters(), lr=1.0)

    std0 = 0.0
    std = 0.0
    std_seed = 0.25
    std_decay_ratio = 0.99

    mean0 = torch.zeros(batch_size, hidden_dim)
    mean_seed = torch.zeros(batch_size, seed_dim)

#    Ptest_cla=test_data.Pdata_cla.cpu().data.numpy()
#    Ptest=test_data.Pdata.cpu().data.numpy()

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

        loss_sum = []
        Pred_cla = []
        Pred_reg = []
        loss_AE_test_sum = 0
        loss_pre_cla_test_sum = 0
        loss_pre_reg_test_sum = 0
        loss_gen_test_sum = 0
        loss_real_test_sum = 0
        loss_cri_test_sum = 0

        file_ARAE = save_result_dir2+"/ARAE_smiles.txt"
        file_AE = save_result_dir2+"/AE_smiles.txt"
        fp_ARAE = open(file_ARAE, "w")
        fp_AE = open(file_AE, "w")

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=False, drop_last=False, num_workers=2)
        model.eval()
        for i, data in enumerate(test_loader):

            batch_x, batch_l, batch_p_reg = data
            batch_x = batch_x.to(device)
            batch_l = batch_l.to(device)
#            batch_p_cla = batch_p_cla.to(device)
#            batch_p_cla1 = F.onehot(batch_p_cla)

            batch_p_reg = batch_p_reg.to(device)
            batch_p = batch_p_reg
#            batch_p = np.cat([batch_p_reg, batch_p_cla])

            batch_x2 = batch_x[:, 1:]
            b_size = batch_x.shape[0]

            noise = mean0.to(device)
            out_decoding = model.AE(batch_x, batch_l, batch_p, noise)
            out2 = out_decoding[:, :-1]
            _, out_num_AE = torch.max(out_decoding, 2)
            loss_AE_test = criterion_AE(
                out2.reshape(-1, Nfea), batch_x2.reshape(-1)).data
            loss_AE_test_sum += loss_AE_test*b_size
#            out_decoding_sf=torch.softmax(out_decoding,dim=2)

            Z_real = model.Enc(batch_x, batch_l)
            Prop_pre = model.Pre(Z_real)
            if Ncla > 0:
                Prop_pre_cla = Prop_pre[:, 0:Ncla]
                loss_pre_cla_test = criterion_Pre_cla(
                    Prop_pre_cla, batch_p_cla).data
                Prop_pre_sf = torch.softmax(Prop_pre_cla, dim=1)
                _, Prop_pre_max = torch.max(Prop_pre_cla, dim=1)
                Pred_cla += [Prop_pre_max.cpu().data.numpy()]
                loss_pre_cla_test_sum += loss_pre_cla_test*b_size

            if Nreg > 0:
                Prop_pre_reg = Prop_pre[:, Ncla:]
                loss_pre_reg_test = criterion_Pre_reg(
                    Prop_pre_reg, batch_p_reg).data
                Pred_reg += [Prop_pre_reg.cpu().data.numpy()]
                loss_pre_reg_test_sum += loss_pre_reg_test*b_size

            batch_s = torch.normal(mean=mean_seed, std=std_seed).to(device)
            Z_gen = model.Gen(batch_s)
            D_gen = model.Cri(Z_gen)
            D_real = model.Cri(Z_real)

            loss_gen_test = -D_gen.mean().data
            loss_real_test = D_real.mean().data
            loss_cri_test = (-D_real.mean() + D_gen.mean()).data

            loss_gen_test_sum += loss_gen_test*b_size
            loss_real_test_sum += loss_real_test*b_size
            loss_cri_test_sum += loss_cri_test*b_size

            out_num_ARAE = model.Dec.decoding(Z_gen, batch_p)

            for k in range(0, 2):
                out_string = vec_to_char(batch_x2[k])
                print("real: ", out_string)
                out_string = vec_to_char(out_num_AE[k])
                print("AE  : ", out_string)
            for k in range(0, 10):
                out_string = vec_to_char(out_num_ARAE[k])
                print("ARAE: ", out_string)

            for k in range(0, b_size):
                line_ARAE = vec_to_char(out_num_ARAE[k])+"\n"
                fp_ARAE.write(line_ARAE)
                line_AE = vec_to_char(out_num_AE[k])+"\n"
                fp_AE.write(line_AE)
        fp_ARAE.close()
        fp_AE.close()

        loss_AE_test = loss_AE_test_sum/Ntest
        loss_pre_cla_test = loss_pre_cla_test_sum/Ntest
        loss_pre_reg_test = loss_pre_reg_test_sum/Ntest
        loss_gen_test = loss_gen_test_sum/Ntest
        loss_real_test = loss_real_test_sum/Ntest
        loss_cri_test = loss_cri_test_sum/Ntest

        acc, acc2 = accu(out_num_AE, batch_x2, batch_l)

        line_out = "%d test: AE %6.3f gen %6.3f cri %6.3f real %6.3f pre_cla %6.3f pre_reg %6.3f " % (
                epoch,
                loss_AE_test, loss_gen_test, loss_cri_test, loss_real_test,
                loss_pre_cla_test, loss_pre_reg_test)
        print(line_out)

        et = time.time()
        print("time: %10.2f" % (et-st))

    print('Finished Training')
    total_et = time.time()
    print("time : %10.2f" % (total_et-total_st))


if __name__ == "__main__":
    main()
