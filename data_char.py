#!/usr/bin/env python
import numpy as np
import os
import sys
import time
from ZINC.char import char_list, char_dict


def gen_data(data_name="train"):
    batch_size = 100

    sample_size = 100
    seq_length = 109

    data_dir = './ZINC'
    data_dir2 = "./data/"
    if not os.path.exists(data_dir2):
        os.makedirs(data_dir2)

    smiles_filename = data_dir+"/"+data_name+"_5.txt"

    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in char_list:
        if len(key) == 1:
            char_list1 += [key]
            char_dict1[key] = char_dict[key]
        elif len(key) == 2:
            char_list2 += [key]
            char_dict2[key] = char_dict[key]
        else:
            print("strange ", key)

    Nchar = len(char_list)

    fp = open(smiles_filename)
    data_lines = fp.readlines()
    fp.close()

    smiles_list = []

    Maxsmiles = 0
    Xdata = []
    Ldata = []
    Pdata = []
    title = ""
    for line in data_lines:
        if line[0] == "#":
            title = line[1:-1]
            title_list = title.split()
            print(title_list)
            continue
        arr = line.split()
        if len(arr) < 2:
            continue
        smiles = arr[0]
        if len(smiles) > seq_length:
            continue
        smiles0 = smiles.ljust(seq_length, '>')
        smiles_list += [smiles]
        Narr = len(arr)
        cdd = []
        for i in range(1, Narr):
            if title_list[i] == "logP":
                cdd += [float(arr[i])/10.0]
            elif title_list[i] == "SAS":
                cdd += [float(arr[i])/10.0]
            elif title_list[i] == "QED":
                cdd += [float(arr[i])/1.0]
            elif title_list[i] == "MW":
                cdd += [float(arr[i])/500.0]
            elif title_list[i] == "TPSA":
                cdd += [float(arr[i])/150.0]

        Pdata += [cdd]  # affinity classification

        X_smiles = '<'+smiles
        X_d = np.zeros([seq_length+1], dtype=int)
        X_d[0] = char_dict['<']

        Nsmiles = len(smiles)
        if Maxsmiles < Nsmiles:
            Maxsmiles = Nsmiles
        i = 0
        istring = 0
        check = True
        while check:
            char2 = smiles[i:i+2]
            char1 = smiles[i]
            if char2 in char_list2:
                j = char_dict2[char2]
                i += 2
                if i >= Nsmiles:
                    check = False
            elif char1 in char_list1:
                j = char_dict1[char1]
                i += 1
                if i >= Nsmiles:
                    check = False
            else:
                print(char1, char2, "error")
                sys.exit()
            X_d[istring+1] = j
            istring += 1
        for i in range(istring, seq_length):
            X_d[i+1] = char_dict['>']

        Xdata += [X_d]
        Ldata += [istring+1]

    Xdata = np.asarray(Xdata, dtype="long")
    Ldata = np.asarray(Ldata, dtype="long")
    Pdata = np.asarray(Pdata, dtype="float32")
    print(Xdata.shape, Ldata.shape, Pdata.shape)

    Xfile = data_dir2+"X"+data_name+".npy"
    Lfile = data_dir2+"L"+data_name+".npy"
    Pfile = data_dir2+"P"+data_name+".npy"
    np.save(Xfile, Xdata)
    np.save(Lfile, Ldata)
    np.save(Pfile, Pdata)


def main():
    gen_data(data_name="train")
    gen_data(data_name="test")


if __name__ == "__main__":
    main()
