#!/usr/bin/env python
import pandas
import numpy as np
from helper import *


[seq_hash, seq, seq_cdr] = make_Sequence_Hash(
    'data/CDR_library_July_5_2013_sequences.txt')

#CDR1 and 3 concatenated
CDRs = 'TFSDYWMNWVGSYYGMDYWG'

#Raw pdb files
CDR_atoms = pandas.read_csv('./data/CDR13_positions.txt',  delimiter=r"\s+", header=None)

#coordinates of sub-data sets
CDR_atoms = CDR_atoms.loc[CDR_atoms[5]==np.round(CDR_atoms[5])]

CDR_atoms = CDR_atoms.loc[~CDR_atoms[2].isin(['C','O','N'])]
#CDR_atoms = CDR_atoms.loc[(~CDR_atoms[2].isin(['CA'])) | (CDR_atoms[3]=='GLY')]
CDRcoords = np.array([CDR_atoms[6].tolist(), CDR_atoms[7].tolist(), CDR_atoms[8].tolist()]).T
CDR_index = CDR_atoms[5]
distances = np.zeros((len(set(CDR_index)),len(set(CDR_index))))
for ii, aa1 in enumerate(np.unique(CDR_index)):
    for jj, aa2 in enumerate(np.unique(CDR_index)):
        usethis1 = np.array(CDR_index==aa1)
        usethis2 = np.array(CDR_index==aa2)
        temp = [np.sqrt(((CDRcoords[usethis2] - aa)**2).sum(axis=1)).min() for aa in CDRcoords[usethis1]]
        distances[ii,jj] = np.min(temp)
