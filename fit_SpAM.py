import sys
import os
import random
import argparse
import logging
import pickle
import subprocess

import numpy as np
from common import write_data_pkl_to_csv

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='seed',
        default=1)
    parser.add_argument('--kfold',
        type=int,
        default=3)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--out-file',
        type=str,
        default="_output/out.csv")
    parser.add_argument('--data-classes',
        type=int,
        default=0)
    parser.add_argument('--num-splines',
        type=int,
        default=8)
    parser.add_argument('--scratch',
        type=str,
        default="_output/scratch")

    args = parser.parse_args()
    return args

def fit_spam(
        args,
        train_file,
        test_file):
    cmd = [
        'Rscript',
        '../R/fit_SpAM.R',
        train_file,
        test_file,
        args.out_file,
        args.data_classes,
        args.seed,
        args.kfold,
        args.num_splines,
    ]
    cmd = list(map(str, cmd))

    print("Calling:", " ".join(cmd))
    res = subprocess.call(cmd)
    # Check that process complete successfully
    assert res == 0

def main(args=sys.argv[1:]):
    args = parse_args()

    randnum = random.randint(1, 100000)
    train_file_name = os.path.join(args.scratch, "train%d" % randnum)
    test_file_name = os.path.join(args.scratch, "test%d" % randnum)
    write_data_pkl_to_csv(
            args.data_file,
            train_file_name,
            test_file_name)

    fit_spam(args, train_file_name, test_file_name)
    os.remove(train_file_name)
    os.remove(test_file_name)

if __name__ == "__main__":
    main(sys.argv[1:])

