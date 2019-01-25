import sys
import time
import argparse
import pickle
import tensorflow as tf
import numpy as np

from data_generator import DataGenerator
from common import *
import data_gen_funcs

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='seed',
        default=1)
    parser.add_argument('--func-name',
        type=str,
        help='name of function from data_gen_funcs.py',
        default='six_variable_multivar_func')
    parser.add_argument('--out-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--n-train',
        type=int,
        default=400)
    parser.add_argument('--n-test',
        type=int,
        default=400)
    parser.add_argument('--num-p',
        type=int,
        default=50)
    parser.add_argument('--snr',
        type=float,
        default=2)

    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    print(args)

    data_gen = DataGenerator(
            args.num_p,
            getattr(data_gen_funcs, args.func_name),
            data_gen_funcs.CLASSIFICATION_DICT[args.func_name],
            snr=args.snr)
    train_data = data_gen.create_data(args.n_train)
    test_data = data_gen.create_data(args.n_test)

    print("data_file %s" % args.out_file)
    with open(args.out_file, "wb") as f:
        pickle.dump({
            "train": train_data,
            "test": test_data},
            f)

if __name__ == "__main__":
    main(sys.argv[1:])
