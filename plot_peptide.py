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
from extractor import amino_acids

HLA_DICT = {
        "A": "HLA-A_01:01",
        "B8": "HLA-B_08:02",
        "B44": "HLA-B_44:02"}

FEATURE_DEFS = ["K%d" % i for i in range(10)] + amino_acids + ["hydropathy", "mass", "aromatic"]
NUM_FEATS = len(FEATURE_DEFS)

def parse_args(args):
    parser = argparse.ArgumentParser("Plot peptide results")
    parser.add_argument(
        '--result-folder',
        type=str,
        default="peptide_binding/_output")
    parser.add_argument(
        '--hla',
        type=str,
        default="A")
    parser.add_argument(
        '--spinn-file-template',
        type=str,
        default="extractor_1/prop_0.20/seed_%d/relu_0/fitted_spinn.pkl")
    parser.add_argument(
        '--nn-file-template',
        type=str,
        default="extractor_1/prop_0.20/seed_%d/relu_0/fitted_%s.csv")
        #default="extractor_1/prop_0.20/seed_%d/relu_0/fitted_%s_final.csv")
    parser.add_argument(
        '--file-template',
        type=str,
        default="extractor_1/prop_0.20/seed_%d/fitted_%s.csv")
    parser.add_argument(
        '--methods',
        type=str,
        default="lasso,trees,spam,spinn,ridge_nn")
    parser.add_argument(
        '--seeds',
        type=str,
        default=make_params(range(3,23)))
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
                res_file = os.path.join(
                    args.result_folder,
                    HLA_DICT[args.hla],
                    args.file_template % (
                        seed,
                        method))
            else:
                res_file = os.path.join(
                    args.result_folder,
                    HLA_DICT[args.hla],
                    args.nn_file_template % (seed, method))
            try:
                results = read_method_result(res_file, method)
            except:
                print('nope')
                continue
            for method_str, mse, num_nonzero in results:
                all_results["method"].append(method_str)
                all_results["mse"].append(mse)
                all_results["num_nonzero"].append(num_nonzero)
                all_results["seed"].append(seed)

    results_df = pd.DataFrame(all_results)
    print(results_df)
    pivot_df = results_df.pivot(index='seed', columns='method', values='mse')
    if 'spam' in args.methods:
        pivot_df = pivot_df.loc[pivot_df['spam'] < 1]
    pivot_agg = pivot_df.agg(['mean', 'var', get_SE])
    print(pivot_agg)
    print(pivot_df)

    pivot_df = results_df.pivot(index='seed', columns='method', values='num_nonzero')
    pivot_agg = pivot_df.agg(['mean', 'var', get_SE])
    print(pivot_agg)
    print(pivot_df)

def get_SE(x):
    return np.sqrt(np.var(x)/x.size)

def plot_weights(args):
    input_counts = {i: 0 for i in range(297)}
    for seed in args.seeds:
        res_file = os.path.join(
            args.result_folder,
            HLA_DICT[args.hla],
            args.spinn_file_template % seed)
        with open(res_file, "rb") as f:
            spinn_res = pickle.load(f)

        num_p = spinn_res["model_params"].coefs[0].shape[0]
        norm_nonzero_inputs = [
            np.linalg.norm(spinn_res["model_params"].coefs[0][i,:], ord=1)
            for i in range(num_p)
        ]
        nonzero_which = np.where(norm_nonzero_inputs)[0]
        for nonzero_input in nonzero_which:
            input_counts[nonzero_input] += 1

    input_count_vals = [v for v in list(input_counts.values()) if v > 0]
    thres_val = np.percentile(input_count_vals, 80)
    print("num times", np.sum([v > thres_val for v in input_count_vals]))
    for k, v in input_counts.items():
        position = int(k/NUM_FEATS)
        if v >= thres_val:
            print(position, FEATURE_DEFS[k % NUM_FEATS])

def main(args=sys.argv[1:]):
    args = parse_args(args)
    plot_mse(args)
    #plot_weights(args)


if __name__ == "__main__":
    main()
