import sys
import argparse
import pickle
import tensorflow as tf
import numpy as np
import logging as log
from multiprocessing import Pool
import time
from sklearn.model_selection import train_test_split

from data_generator import Dataset
from settings import *
from common import *
from neural_network import NeuralNetwork
from spinn_common import run_simulation

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('data',
        type=str,
        help="File path with the data in csv format. Last column is y, rest of the columns are X")
    parser.add_argument('--seeds',
        type=str,
        help='Comma-separated list of seeds',
        default="1")
    parser.add_argument('--num-threads',
        type=int,
        help='Number of threads',
        default=1)
    parser.add_argument('--outdir',
        type=str,
        default="_output/",
        help="Output directory")
    parser.add_argument('--lasso-param-ratios',
        type=str,
        default="0.1",
        help="""
            ratio between lasso penalty parameters and group lasso penalty parameters, comma separated.
            e.g. 0.1,0.01 means that we will test all lasso penalty parameters that are 0.1 times and
            0.01 times the group lasso penalty parameters.
            """)
    parser.add_argument('--group-lasso-params',
        type=str,
        default="0.1",
        help="group lasso penalty parameters, comma separated.")
    parser.add_argument('--ridge-params',
        type=str,
        default='0.0001',
        help="ridge penalty parameters, comma separated.")
    parser.add_argument('--hidden-sizes',
        type=str,
        default="3",
        help="comma-separated list of number of hidden nodes")
    parser.add_argument('--kfold',
        type=int,
        default=5,
        help="number of folds for cross validation")
    args = parser.parse_args()
    args.ridge_params = process_params(args.ridge_params, float)
    args.lasso_param_ratios = process_params(args.lasso_param_ratios, float)
    args.group_lasso_params = process_params(args.group_lasso_params, float)
    args.hidden_sizes = process_params(args.hidden_sizes, int)

    args.seed_list = process_params(args.seeds, int)
    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    X, y = read_data(args.data, has_header=True)
    y -= np.mean(y)

    pool = Pool(args.num_threads) if args.num_threads > 1 else None
    args.num_p = X.shape[1]
    args.n_data = X.shape[0]
    proportion = 1 - 1.0/args.kfold if args.kfold > 0 else VAL_PROP_1FOLD
    val_proportion = VAL_PROP_1FOLD
    args.n_train = int(args.n_data * (1 - 1.0/args.kfold)) if args.kfold > 0 else int(args.n_data * proportion)
    args.log_file = make_log_file_name(args)
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    log.info(args)

    pool = Pool(args.num_threads) if args.num_threads > 1 else None
    test_losses = []
    for seed in args.seed_list:
        shuffled_idx = np.random.choice(args.n_data, size=args.n_data, replace=False)
        X = X[shuffled_idx, :]
        y = y[shuffled_idx]
        train_idx = int(args.n_data * proportion)

        X_train = X[:train_idx,:]
        X_test = X[train_idx:,:]
        y_train = y[:train_idx]
        y_test = y[train_idx:]

        if args.kfold > 0:
            dataset = Dataset(
                x_train=X_train,
                y_train=y_train.reshape((y_train.size, 1)),
                x_test=X_test,
                y_test=y_test.reshape((y_test.size, 1)),
                y_test_true=y_test.reshape((y_test.size, 1)),
            )
        else:
            val_idx = int(X_train.shape[0] * val_proportion)
            X_train_train = X_train[:val_idx,:]
            X_val = X_train[val_idx:,:]
            y_train_train = y_train[:val_idx]
            y_val = y_train[val_idx:]

            dataset = Dataset(
                x_train=X_train_train,
                y_train=y_train_train.reshape((y_train_train.size, 1)),
                x_val=X_val,
                y_val=y_val.reshape((y_val.size, 1)),
                x_test=X_test,
                y_test=y_test.reshape((y_test.size, 1)),
                y_test_true=y_test.reshape((y_test.size, 1)),
            )

        settings = Settings(
            nn_cls = NeuralNetwork,
            func = None,
            n_train = y_train.size,
            n_val = 0,
            n_test = y_test.size,
            num_p = X_train.shape[1],
            snr = None,
            ridge_param = None,
            lasso_param = None,
            group_lasso_param = None,
            hidden_sizes = None,
            num_inits = 2,
            max_iters = 2000,
        )
        best_test_loss = run_simulation(seed, args, settings, pool, dataset=dataset, kfold=args.kfold, shuffled_idx=shuffled_idx)
        test_losses.append(best_test_loss)
        log.info("FINAL seed %d, Best Test loss %f" % (seed, best_test_loss))

    log.info("AVG Test loss %s" % get_summary(test_losses))

    if pool is not None:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main(sys.argv[1:])
