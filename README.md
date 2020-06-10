# ARAE torch for SMILES generation


# Prerequisite:

python3

numpy

pytorch 

RDKit

SA_Score for RDKit

https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score

# Download:

git clone https://github.com/gicsaw/ARAE_torch


# Training data preparation :
cd ARAE_torch

### generate data

python data_char.py

### trainning (skip)

unconditional

python ARAE_train.py 

conditional 

python CARAE_train.py 

### generation 

unconditional

python ARAE_gen.py 

conditional

python CARAE_gen.py $logP $MW $QED

### valid, unique, and novel check

export PYTHONPATH="SA_Score directory":${PYTHONPATH}

python valid.py result_CARAE_gen/epoch59

output file: result_CARAE_gen/epoch59/smiles_novel.txt
