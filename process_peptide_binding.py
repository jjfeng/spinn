"""
Run `grep` on the appropriate lines from
mhcflurry/4/1.2.0/data_curated/curated_training_data.no_mass_spec.csv
to create a csv file to make a file like "../data/HLA-A_01:01.csv".

This will output a new file "../data/HLA-A_01:01_processed.csv"
Input this into `read_data.py`
"""

import os
import sys
import csv

import numpy as np
import pandas as pd

from extractor import Extractor
from Bio import SeqIO

file_name = "../data/HLA-A_01:01.csv"
#file_name = "../data/HLA-B_08:02.csv"
#file_name = "../data/HLA-B_44:02.csv"

feat_maker = Extractor()

Xs = []
ys = []
with open(file_name, "r") as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)
    for line in csv_reader:
        peptide = line[1]
        val = float(line[2])
        ineq = line[3]
        if ineq != "=":
            #print(line)
            continue
        if len(peptide) != 9:
            continue

        x = feat_maker.extract(peptide)
        Xs.append(x)
        ys.append(val)

Xs = np.array(Xs)
ys = np.array(ys).reshape(-1,1)
ys = np.log(ys)
print(ys.max())
print(ys.min())
print(np.median(ys))
print(np.mean(ys))
print(np.var(ys))
out_data = np.concatenate([Xs, ys], axis=1)
print(out_data.shape)
np.savetxt(
        file_name.replace(".csv", "_processed.csv"), out_data, delimiter=",",
        header=",".join(["x%d" % i for i in range(out_data.shape[0] - 1)] + ["y"]))
