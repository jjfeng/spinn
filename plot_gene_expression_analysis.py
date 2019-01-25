import sys
import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import csv

from plot_simulation_general import read_method_result
from common import process_params, make_params

def parse_args(args):
    parser = argparse.ArgumentParser("Plot riboflavin")
    parser.add_argument(
        '--result-folder',
        type=str,
        default="riboflavin")
    parser.add_argument(
        '--spinn-file-template',
        type=str,
        default="extractor_1/prop_0.20/seed_%d/relu_0/fitted_spinn.pkl")
    parser.add_argument(
        '--nn-file-template',
        type=str,
        #default="_output/seed_%d/relu_0/layer_10,10:2,10:4/fitted_%s.csv")
        default="_output/seed_%d/relu_0/layer_3,10/fitted_%s.csv")
    parser.add_argument(
        '--file-template',
        type=str,
        default="_output/seed_%d/fitted_%s.csv")
    parser.add_argument(
        '--methods',
        type=str,
        default="lasso,trees,spinn,spam,ridge_nn")
    parser.add_argument(
        '--seeds',
        type=str,
        default=make_params(range(40,70)))
    parser.set_defaults()
    args = parser.parse_args(args)
    args.methods = args.methods.split(",")
    args.seeds = process_params(args.seeds, int)
    return args

def plot_mse(args):
    all_results = {
            "method": [],
            "mse": [],
            "num_nonzero": [],
            "seed": []}
    for seed in args.seeds:
        for method in args.methods:
            if method not in ["spinn", "ridge_nn"]:
                res_file = os.path.join(args.result_folder,
                    args.file_template % (
                        seed,
                        method))
            else:
                res_file = os.path.join(
                    args.result_folder,
                    args.nn_file_template % (seed, method))
            try:
                results = read_method_result(res_file, method)
                for method_str, mse, num_nonzero in results:
                    all_results["method"].append(method_str)
                    all_results["mse"].append(float(mse))
                    all_results["num_nonzero"].append(num_nonzero)
                    all_results["seed"].append(seed)
            except:
                print("nope", res_file)

    results_df = pd.DataFrame(all_results)
    pivot_df = results_df.pivot(index='seed', columns='method', values='mse')
    print(pivot_df)
    #if 'spinn' in args.methods:
    #    pivot_df = pivot_df.loc[pivot_df['spinn'] < .4]
    #if 'spam' in args.methods:
    #    pivot_df = pivot_df.loc[pivot_df['spam'] < 1]
    pivot_agg = pivot_df.agg(['mean', 'var', get_SE])
    print(pivot_agg)

    pivot_df = results_df.pivot(index='seed', columns='method', values='num_nonzero')
    pivot_agg = pivot_df.agg(['mean', 'var', get_SE])
    print(pivot_agg)

def get_SE(x):
    return np.sqrt(np.var(x)/x.size)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    plot_mse(args)


if __name__ == "__main__":
    main()
