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
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--out-file',
        type=str,
        default="_output/restrict_data.pkl")
    parser.add_argument('--max-relevant-idx',
        type=int,
        default=6)
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    print(args)

    with open(args.data_file, "rb") as f:
        all_data = pickle.load(f)
        train_data = all_data["train"]
        test_data = all_data["test"]

    print("data_file %s" % args.out_file)
    with open(args.out_file, "wb") as f:
        pickle.dump({
            "train": train_data.create_restricted(args.max_relevant_idx),
            "test": test_data.create_restricted(args.max_relevant_idx),
            }, f)

if __name__ == "__main__":
    main(sys.argv[1:])
