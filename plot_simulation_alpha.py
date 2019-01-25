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

from common import process_params, make_params, THRES
from plot_simulation_general import read_method_result

def parse_args(args):
    parser = argparse.ArgumentParser("Plot for the main simulations")
    parser.add_argument(
        '--result-folder',
        type=str,
        default="simulation_alpha")
    parser.add_argument(
        '--lasso',
        type=str,
        default=make_params(np.arange(-.5, -6, step=-.5)))
    parser.add_argument(
        '--group-lasso',
        type=str,
        default=make_params(np.arange(-.5, -6, step=-.5)))
    parser.add_argument(
        '--max-relevant-idx',
        type=int,
        default=6)
    parser.add_argument(
        '--seeds',
        type=str,
        default="4,5,6,7,8,9,10,11")
    parser.add_argument(
        '--file-template',
        type=str,
        default="_output/seed_%d/group_lasso_%.2f/lasso_%.2f/fitted_spinn.%s")
    parser.add_argument(
        '--out-mse-plot',
        type=str,
        default="_output/plot_alpha_mse.png")
    parser.add_argument(
        '--out-irrelev-weight-plot',
        type=str,
        default="_output/plot_alpha_irrelev_weight.png")
    parser.add_argument(
        '--out-relev-weight-plot',
        type=str,
        default="_output/plot_alpha_relev_weight.png")
    parser.add_argument(
        '--out-nonzero-hidden-plot',
        type=str,
        default="_output/plot_alpha_nonzero_hidden.png")
    parser.add_argument(
        '--out-nonzero-inputs-plot',
        type=str,
        default="_output/plot_alpha_nonzero_inputs.png")
    parser.set_defaults()
    args = parser.parse_args(args)
    args.group_lasso = process_params(args.group_lasso, float)
    args.lasso = process_params(args.lasso, float)
    args.seeds = process_params(args.seeds, int)
    return args

def plot_mse(args):
    all_results = {
            "lasso": [],
            "group_lasso": [],
            "mse": []}
    for lasso in args.lasso:
        for group_lasso in args.group_lasso:
            mses = []
            for seed in args.seeds:
                res_file = os.path.join(args.result_folder, args.file_template % (
                    seed,
                    group_lasso,
                    lasso,
                    "csv"))
                _, mse, _ = read_method_result(res_file, "spinn")[0]
                mses.append(float(mse))
            all_results["lasso"].append(lasso)
            all_results["group_lasso"].append(group_lasso)
            all_results["mse"].append(np.mean(mses))

    results_df = pd.DataFrame(all_results)

    plt.clf()
    plt.figure(figsize=(3.8,3.25))
    pivot_df = results_df.pivot("lasso", "group_lasso", "mse")
    print("MSE")
    print(np.log(pivot_df))
    sns_plt = sns.heatmap(np.log(pivot_df))
    plt.xlabel("$\log_{10}$(Group lasso parameter)")
    plt.ylabel("$\log_{10}$(Lasso parameter)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_mse_plot))

def plot_weights(args):
    all_results = {
            "lasso": [],
            "group_lasso": [],
            "relevant": [],
            "weight": [],
    }
    for lasso in args.lasso:
        for group_lasso in args.group_lasso:
            relev_norms = []
            irrelev_norms = []
            for seed in args.seeds:
                res_file = os.path.join(args.result_folder, args.file_template % (
                    seed,
                    group_lasso,
                    lasso,
                    "pkl"))
                with open(res_file, "rb") as f:
                    spinn_res = pickle.load(f)

                num_p = spinn_res["model_params"].coefs[0].shape[0]
                total_layer_norm = np.sum(np.abs(spinn_res["model_params"].coefs[0]))
                #print(total_layer_norm)
                norm_nonzero_inputs = [
                    np.linalg.norm(spinn_res["model_params"].coefs[0][i,:], ord=1)/total_layer_norm
                    if total_layer_norm > 0 else 0
                    for i in range(num_p)
                ]
                relev_norms.append(
                    np.sum(norm_nonzero_inputs[:args.max_relevant_idx]))
                irrelev_norms.append(
                    np.sum(norm_nonzero_inputs[args.max_relevant_idx:]))

            results = [
                    [True, np.mean(relev_norms)],
                    [False, np.mean(irrelev_norms)],
            ]
            for relevant, mean_weight in results:
                all_results["relevant"].append(relevant)
                all_results["weight"].append(mean_weight)
                all_results["lasso"].append(lasso)
                all_results["group_lasso"].append(group_lasso)

    results_df = pd.DataFrame(all_results)

    plt.clf()
    plt.figure(figsize=(3.85,3.25))
    irrev_results_df = results_df.loc[results_df["relevant"] == False,:]
    pivot_df= irrev_results_df.pivot("lasso", "group_lasso", "weight")
    print(pivot_df)
    sns_plt = sns.heatmap(pivot_df)
    plt.xlabel("$\log_{10}$(Group lasso parameter)")
    plt.ylabel("$\log_{10}$(Lasso parameter)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_irrelev_weight_plot))

    plt.clf()
    plt.figure(figsize=(4,3.25))
    relev_results_df = results_df.loc[results_df["relevant"],:]
    pivot_df= relev_results_df.pivot("lasso", "group_lasso", "weight")
    print(pivot_df)
    sns_plt = sns.heatmap(pivot_df)
    plt.xlabel("$\log_{10}$(Group lasso parameter)")
    plt.ylabel("$\log_{10}$(Lasso parameter)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_relev_weight_plot))

def plot_nonzero_count(args):
    all_results = {
            "lasso": [],
            "group_lasso": [],
            "num_nonzero_per_hidden": [],
            "num_nonzero_inputs": [],
    }
    for lasso in args.lasso:
        for group_lasso in args.group_lasso:
            res_file = os.path.join(args.result_folder, args.file_template % (
                group_lasso,
                lasso,
                "pkl"))
            with open(res_file, "rb") as f:
                spinn_res = pickle.load(f)

            num_p = spinn_res["model_params"].coefs[0].shape[0]
            nonzero_per_hidden = np.sum(np.abs(spinn_res["model_params"].coefs[0]) >= THRES, axis=0)
            #print("nonz", nonzero_per_hidden, np.median(nonzero_per_hidden))
            all_results["num_nonzero_per_hidden"].append(float(np.median(nonzero_per_hidden)))
            nonzero_inputs = np.sum(np.abs(spinn_res["model_params"].coefs[0]), axis=1) >= THRES
            #print("num nonzero inpu", np.sum(nonzero_inputs))
            all_results["num_nonzero_inputs"].append(np.sum(nonzero_inputs))
            all_results["lasso"].append(lasso)
            all_results["group_lasso"].append(group_lasso)

    results_df = pd.DataFrame(all_results)
    print(all_results["num_nonzero_per_hidden"])
    results_df["percent_nonzero_per_hidden"] = results_df["num_nonzero_per_hidden"]/(results_df["num_nonzero_inputs"] + 1e-10)

    plt.clf()
    pivot_df = results_df.pivot("lasso", "group_lasso", "num_nonzero_inputs")
    print(pivot_df)
    sns_plt = sns.heatmap(pivot_df)
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_nonzero_inputs_plot))

    plt.clf()
    pivot_df= results_df.pivot("lasso", "group_lasso", "percent_nonzero_per_hidden")
    print(pivot_df)
    sns_plt = sns.heatmap(pivot_df)
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_nonzero_hidden_plot))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    plot_mse(args)
    plot_weights(args)
    #plot_nonzero_count(args)


if __name__ == "__main__":
    main()
