#!/usr/bin/env python
import sys, os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem import MolStandardize
#from molvs import tautomer
from rdkit import DataStructs

from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue

import sascorer

USAGE="""
valid.py data_dir
"""

def creator(q, data, Nproc):
    Ndata = len(data)
    for d in data:
        idx=d[0]
        smiles=d[1]
        q.put((idx, smiles))

    for i in range(0, Nproc):
        q.put('DONE')


def check_validity(q,return_dict_valid):

    while True:
        qqq = q.get()
        if qqq == 'DONE':
#            print('proc =', os.getpid())
            break
        idx, smi0 = qqq

        index=smi0.find('>')
        smi=smi0[0:index]

        if idx%10000==0:
            print(idx)

        m=Chem.MolFromSmiles(smi)
        if m == None :
            continue
        if Chem.SanitizeMol(m,catchErrors=True):
            continue
        smi2=Chem.MolToSmiles(m)
#        smi2=MolStandardize.canonicalize_tautomer_smiles(smi)

        return_dict_valid[idx]=[smi2]

def cal_fp(q,return_dict_fp):

    nbits=1024
    while True:
        qqq = q.get()
        if qqq == 'DONE':
#            print('proc =', os.getpid())
            break
        idx, smi = qqq

        if idx%10000==0:
            print(idx)
        Nsmi=len(smi)
        mol=Chem.MolFromSmiles(smi)
        if mol == None :
            continue
        if Chem.SanitizeMol(mol,catchErrors=True):
            continue

        com_fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=nbits)
        return_dict_fp[idx]=[com_fp]

def cal_sim(q,ref_data,return_dict_sim):

    Nref=len(ref_data)
    nbits=1024
    while True:
        qqq = q.get()
        if qqq == 'DONE':
#            print('proc =', os.getpid())
            break
        idx, smi = qqq

        if idx%10000==0:
            print(idx)
        Nsmi=len(smi)
        mol=Chem.MolFromSmiles(smi)
        if mol == None :
            continue
        if Chem.SanitizeMol(mol,catchErrors=True):
            continue

        com_fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=nbits)
        sim_data=[]
        for j in range(Nref):
            ref_fp=ref_data[j][1]
            sim=DataStructs.TanimotoSimilarity(com_fp,ref_fp)
            sim_data+=[sim]
        similarity=np.array(sim_data)
        j_max=similarity.argmax()
        sim_max=similarity[j_max]
        return_dict_sim[idx]=[sim_max,j_max]

def cal_prop(q,return_dict_prop):

    nbits=1024
    while True:
        qqq = q.get()
        if qqq == 'DONE':
#            print('proc =', os.getpid())
            break
        idx, smi = qqq

#        if idx%10000==0:
#            print(idx)
        mol = Chem.MolFromSmiles(smi)
        logP = MolLogP(mol)
        SAS = sascorer.calculateScore(mol)
        QED = qed(mol)
        MW = MolWt(mol)
        TPSA0 = TPSA(mol)

        return_dict_prop[idx]=[logP,SAS,QED,MW,TPSA0]

def main():
    if len(sys.argv)<1:
        print(USAGE)
        sys.exit()

    data_dir=sys.argv[1]

    Nproc=30
    gen_file=data_dir+"/ARAE_smiles.txt"
    fp=open(gen_file)
    lines=fp.readlines()
    fp.close()
    k=-1
    gen_data=[]
    for line in lines:
        if line.startswith("#"):
            continue
        k+=1
        smi=line.strip()
        gen_data+=[[k,smi]]

    Ndata=len(gen_data)

    q = Queue()
    manager = Manager()
    return_dict_valid = manager.dict()
    proc_master = Process(target=creator, args=(q, gen_data, Nproc))
    proc_master.start()

    procs = []
    for k in range(0, Nproc):
        proc = Process(target=check_validity, args=(q,return_dict_valid))
        procs.append(proc)
        proc.start()

    q.close()
    q.join_thread()
    proc_master.join()
    for proc in procs:
        proc.join()

    keys = sorted(return_dict_valid.keys())
    num_valid=keys

    valid_smi_list=[]
    for idx in keys:
        valid_smi=return_dict_valid[idx][0]
        valid_smi_list+=[valid_smi]

    num_valid=len(valid_smi_list)

    line_out="valid:  %6d %6d %6.4f" %(num_valid,Ndata,float(num_valid)/Ndata)
    print(line_out)

    unique_set=set(valid_smi_list)
    num_set=len(unique_set)
    unique_list=sorted(unique_set)

    line_out="Unique:  %6d %6d %6.4f" %(num_set,num_valid,float(num_set)/float(num_valid))
    print(line_out)

    file_output2=data_dir+"/smiles_unique.txt"
    fp_out2=open(file_output2,"w")
    line_out="#smi\n"
    fp_out2.write(line_out)

    for smi in unique_list:
        line_out="%s\n" %(smi)
        fp_out2.write(line_out)
    fp_out2.close()

    ZINC_file="ZINC/train_5.txt"
    ZINC_data=[x.strip().split()[0] for x in open(ZINC_file) if not x.startswith("SMILES")]
    ZINC_set=set(ZINC_data)
    novel_list=list(unique_set-ZINC_set)

    novel_data=[]
    for idx, smi in enumerate(novel_list):
        novel_data+=[[idx,smi]]

    q2 = Queue()
    manager = Manager()
    return_dict_prop = manager.dict()
    proc_master = Process(target=creator, args=(q2, novel_data, Nproc))
    proc_master.start()

    procs = []
    for k in range(0, Nproc):
        proc = Process(target=cal_prop, args=(q2,return_dict_prop))
        procs.append(proc)
        proc.start()

    q2.close()
    q2.join_thread()
    proc_master.join()
    for proc in procs:
        proc.join()

    num_novel=len(novel_list)

    line_out="Novel:  %6d %6d %6.4f" %(num_novel,num_set,float(num_novel)/float(num_set))
    print(line_out)

    file_output3=data_dir+"/smiles_novel.txt"
    fp_out3=open(file_output3,"w")

    keys=sorted(return_dict_prop.keys())

    for key in keys:
        smi=novel_data[key][1]
        prop=return_dict_prop[key]
        logP, SAS, QED, MW, TPSA = prop
        line_out="%s %6.3f %6.3f %5.3f %7.3f %7.3f\n" %(smi, logP, SAS, QED, MW, TPSA)
        fp_out3.write(line_out)
    fp_out3.close()



if __name__=="__main__":
    main()






