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

from common import process_params, make_params

METHOD_DICT = {
    "spinn": "SPINN",
    "ridge_nn": "Ridge-only NN",
    "oracle_nn": "Oracle NN",
    "spam": "SpAM",
    "lasso": "Lasso",
    "GAM multivariate": "GAM multivariate",
    "GAM univariate": "GAM univariate",
    "trees": "Random forest",
}

def parse_args(args):
    parser = argparse.ArgumentParser("Plot for the main simulations")
    parser.add_argument(
        '--result-folder',
        type=str,
        default="simulation_univar_additive")
    parser.add_argument(
        '--spinn-template',
        type=str,
        #default="_output/seed_%d/n_train_%d/fitted_spinn.pkl")
        default="_output/seed_%d/n_train_%d/fitted_spinn.pkl")
    parser.add_argument(
        '--file-template',
        type=str,
        default="_output/seed_%d/n_train_%d/fitted_%s.csv")
    parser.add_argument(
        '--methods',
        type=str,
        default="spam,lasso,spinn,gam,trees,ridge_nn,oracle_nn")
    parser.add_argument(
        '--max-relevant-idx',
        type=int,
        default=6)
    parser.add_argument(
        '--n-trains',
        type=str,
        #default=make_params([125, 250, 500, 1000]))
        default=make_params([125, 250, 500, 1000, 2000, 4000, 8000]))
    parser.add_argument(
        '--seeds',
        type=str,
        default=make_params(range(2,22)))
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/plot_simulation_mse.png")
    parser.add_argument(
        '--out-weight-plot',
        type=str,
        default="_output/plot_simulation_weights.png")
    parser.add_argument(
        '--show-legend',
        action='store_true')
    parser.set_defaults()
    args = parser.parse_args(args)
    args.methods = args.methods.split(",")
    args.seeds = process_params(args.seeds, int)
    args.n_trains = process_params(args.n_trains, int)
    return args

def read_method_result(res_file, method):
    with open(res_file, "r") as f:
        print(res_file)
        reader = csv.reader(f)
        header = next(reader)
        if method == "trees":
            row = next(reader)
            return [(method, float(row[1]), np.nan)]
        elif method == "gam":
            results = []
            for row in reader:
                results.append((row[0], float(row[1]), 6))
            return results
        elif method == "spam":
            for row in reader:
                if "test" in row[0]:
                    mse = float(row[1])
                elif "zero" in row[0]:
                    num_nonzero = int(float(row[1]))
            return [(method, mse, num_nonzero)]
        elif method == "lasso":
            for row in reader:
                if row[0] == "test_loss" or row[0] == "Lasso" or row[0] == "Lasso test accuracy":
                    mse = float(row[1])
                elif row[0] == "num_nonzero" or row[0] == "Lasso nonzero elems":
                    num_nonzero = int(float(row[1]))
            return [(method, mse, num_nonzero)]
        elif method == "spinn" or method == "spinn_new":
            for row in reader:
                if row[0] == "test_loss":
                    mse = float(row[1])
                elif row[0] == "num_nonzero":
                    num_nonzero = int(float(row[1]))
            return [(method, mse, num_nonzero)]
        elif method == "ridge_nn":
            for row in reader:
                if row[0] == "test_loss":
                    mse = float(row[1])
                elif row[0] == "num_nonzero":
                    num_nonzero = int(float(row[1]))
            return [(method, mse, num_nonzero)]
        elif method == "oracle_nn":
            for row in reader:
                if row[0] == "test_loss":
                    mse = float(row[1])
                elif row[0] == "num_nonzero":
                    num_nonzero = int(float(row[1]))
            return [(method, mse, num_nonzero)]
        else:
            raise ValueError("what method?! %s" % method)

def plot_mse(args):
    all_results = {
            "method": [],
            "mse": [],
            "n_train": [],
            "seed": []}
    for seed in args.seeds:
        for n_train in args.n_trains:
            for method in args.methods:
                res_file = os.path.join(args.result_folder,
                    args.file_template % (
                        seed,
                        n_train,
                        method))
                try:
                    results = read_method_result(res_file, method)
                except:
                    print("cant find", res_file)
                    continue
                for method_str, mse, num_nonzero in results:
                    if method_str == "spinn" and num_nonzero == 0:
                        continue
                    all_results["method"].append(METHOD_DICT[method_str])
                    all_results["mse"].append(float(mse))
                    all_results["seed"].append(seed)
                    all_results["n_train"].append(int(n_train))
    results_df = pd.DataFrame(all_results)
    print("RESULTS")
    pivot_df = results_df[results_df['n_train'] == 500].pivot(index='seed', columns='method', values='mse')
    print(pivot_df)

    plt.clf()
    if args.show_legend:
        plt.figure(figsize=(5,5))
    else:
        plt.figure(figsize=(5,4))
    hue_order = list(sorted(list(set(all_results["method"]))))
    print(hue_order)
    sns_plt = sns.pointplot(
            x="n_train",
            y="mse",
            hue="method",
            data=results_df,
            hue_order=hue_order,
            linestyles=["--" if ("GAM" in h or "Oracle" in h) else "-" for h in hue_order],
            markers=["o" if ("GAM" in h or "Oracle" in h) else "v" for h in hue_order])
    sns_plt.set_yscale('log')
    plt.xlabel("Number of observations")
    plt.ylabel("Mean squared error")
    if args.show_legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        plt.tight_layout()
    else:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    plt.savefig(os.path.join(args.result_folder, args.out_plot))

def plot_weights(args):
    all_results = {
            "relevant": [],
            "weight": [],
            "n_train": [],
            "seed": []}
    for seed in args.seeds:
        for n_train in args.n_trains:
            res_file = os.path.join(args.result_folder,
                args.spinn_template % (
                    seed,
                    n_train))
            with open(res_file, "rb") as f:
                spinn_res = pickle.load(f)

            num_p = spinn_res["model_params"].coefs[0].shape[0]
            norm_nonzero_inputs = [
                np.linalg.norm(spinn_res["model_params"].coefs[0][i,:], ord=2)
                for i in range(num_p)
            ]
            results = [
                    [True, np.sum(norm_nonzero_inputs[:args.max_relevant_idx])],
                    [False, np.sum(norm_nonzero_inputs[args.max_relevant_idx:])],
            ]
            for relevant, mean_weight in results:
                all_results["relevant"].append(relevant)
                all_results["weight"].append(mean_weight)
                all_results["seed"].append(seed)
                all_results["n_train"].append(int(n_train))
    results_df = pd.DataFrame(all_results)
    #print("RESULTS")
    #print(results_df)

    plt.clf()
    plt.figure(figsize=(5,4))
    sns_plt = sns.pointplot(
            x="n_train",
            y="weight",
            hue="relevant",
            data=results_df,
            linestyles=["-", "--"],
            markers=['o','v'])
    sns_plt.set_yscale('log')
    plt.xlabel("Number of observations")
    plt.ylabel("Log (L2 norm of irrelevant weights)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_weight_plot))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    plot_mse(args)
    plot_weights(args)


if __name__ == "__main__":
    main()
