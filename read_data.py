import sys
import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import GEOparse

from data_generator import Dataset
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='seed',
        default=1)
    parser.add_argument('--scale-y',
        action='store_true')
    parser.add_argument('--center-y',
        action='store_true')
    parser.add_argument('--test-proportion',
        type=float,
        default=0.2)
    parser.add_argument('--in-file',
        type=str,
        default="../data/riboflavin.csv")
    parser.add_argument('--out-file',
        type=str,
        default="_output/data.pkl")

    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)
    print(args)

    X, y = read_data(args.in_file, has_header=True)

    print(X.shape)
    print(y.shape)
    print(y.mean())
    y = y.reshape(y.size, 1)
    if args.center_y:
        y -= np.mean(y)
    if args.scale_y:
        y /= np.sqrt(np.var(y))
    shuffled_idx = np.random.choice(y.size, size=y.size, replace=False)
    shuff_X = X[shuffled_idx, :]
    shuff_y = y[shuffled_idx]

    n_train = y.size - int(y.size * args.test_proportion)

    train_data = Dataset(
            shuff_X[:n_train, :],
            shuff_y[:n_train,:],
            shuff_y[:n_train,:])
    test_data = Dataset(
            shuff_X[n_train:, :],
            shuff_y[n_train:,:],
            shuff_y[n_train:,:])

    print("data_file %s" % args.out_file)
    with open(args.out_file, "wb") as f:
        pickle.dump({
            "train": train_data,
            "test": test_data},
            f)

if __name__ == "__main__":
    main(sys.argv[1:])
