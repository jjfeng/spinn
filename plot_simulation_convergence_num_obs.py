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
import statsmodels.api as sm

from plot_simulation_general import read_method_result
from common import process_params, make_params

def parse_args(args):
    parser = argparse.ArgumentParser("Plot for the simulation convergence check")
    parser.add_argument(
        '--result-folder',
        type=str,
        default="simulation_convergence_num_obs")
    parser.add_argument(
        '--spinn-template',
        type=str,
        default="_output/seed_%d/n_train_%d/fitted_spinn.pkl")
    parser.add_argument(
        '--file-template',
        type=str,
        default="_output/seed_%d/n_train_%d/fitted_spinn.csv")
    parser.add_argument(
        '--n-trains',
        type=str,
        default=make_params([100, 200, 400, 800, 1600, 3200]))
    parser.add_argument(
        '--lasso-ratio',
        type=float,
        default=0.1)
    parser.add_argument(
        '--max-relevant-idx',
        type=int,
        default=6)
    parser.add_argument(
        '--seeds',
        type=str,
        default=make_params(range(11,31)))
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/plot_simulation_mse.png")
    parser.add_argument(
        '--out-weight-plot',
        type=str,
        default="_output/plot_simulation_weights.png")
    parser.set_defaults()
    args = parser.parse_args(args)
    args.seeds = process_params(args.seeds, int)
    args.n_trains = process_params(args.n_trains, int)
    return args

def plot_mse(args):
    all_results = {
            "mse": [],
            "n_train": [],
            "seed": []}
    for seed in args.seeds:
        for n_train in args.n_trains:
            res_file = os.path.join(args.result_folder,
                args.file_template % (
                    seed,
                    n_train))
            results = read_method_result(res_file, "spinn")
            for _, mse, _ in results:
                all_results["mse"].append(float(mse))
                all_results["seed"].append(seed)
                all_results["n_train"].append(int(n_train))
    results_df = pd.DataFrame(all_results)
    #print("RESULTS")
    #print(results_df)

    X = np.log(np.log(results_df["n_train"])/results_df["n_train"]).reshape(-1,1)
    y = np.log(results_df["mse"])
    ols = sm.OLS(y, sm.add_constant(X))
    ols_results = ols.fit()
    print(ols_results.summary())
    print("covariance", ols_results.cov_HC0)
    print('Parameters: ', ols_results.params)
    print('R2: ', ols_results.rsquared)

    plt.clf()
    results_df["inv_n_train"] = np.log(results_df["n_train"])/results_df["n_train"]
    sns_plt = sns.regplot(
            x="inv_n_train",
            y="mse",
            data=results_df,
            x_jitter=0.001,
            scatter_kws={'alpha':0.3},
            lowess=True)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("Log n / n", size=16)
    plt.ylabel("Excess loss", size=16)
    plt.tight_layout()
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
                (args.lasso_ratio * np.linalg.norm(spinn_res["model_params"].coefs[0][i,:], ord=1)
                + np.linalg.norm(spinn_res["model_params"].coefs[0][i,:], ord=2))
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

    irrev_results_df = results_df.loc[results_df["relevant"] == False,:]
    lm_results_df = irrev_results_df.loc[irrev_results_df["n_train"] > 200,:]
    X = np.log(np.log(lm_results_df["n_train"])/lm_results_df["n_train"]).reshape(-1,1)
    y = np.log(lm_results_df["weight"])
    ols = sm.OLS(y, sm.add_constant(X))
    ols_results = ols.fit()
    print(ols_results.summary())
    print("covariance", ols_results.cov_HC0)
    print('Parameters: ', ols_results.params)
    print('R2: ', ols_results.rsquared)

    plt.clf()
    irrev_results_df["inv_sqrt_n_train"] = np.sqrt(np.log(irrev_results_df["n_train"])/results_df["n_train"])
    sns_plt = sns.regplot(
            x="inv_sqrt_n_train",
            y="weight",
            data=irrev_results_df,
            x_jitter=0.001,
            scatter_kws={'alpha':0.3},
            lowess=True)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("Sqrt(Log n / n)", size=16)
    plt.ylabel("SGL of irrelevant weights", size=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_folder, args.out_weight_plot))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    plot_mse(args)
    plot_weights(args)


if __name__ == "__main__":
    main()

