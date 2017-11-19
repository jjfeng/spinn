import sys
import argparse
import pickle
import tensorflow as tf
import numpy as np
import logging as log
from multiprocessing import Pool
import time

from data_generator import DataGenerator
from data_generator import six_variable_func
from settings import *
from common import *
from neural_network import NeuralNetwork
from spinn_common import run_simulation

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

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
        default="_output/")
    parser.add_argument('--lasso-param-ratios',
        type=str,
        default="0.1",
        help="""
            Ratio between lasso penalty parameters and group lasso penalty parameters, comma separated.
            e.g. 0.1,0.01 means that we will test all lasso penalty parameters that are 0.1 times and
            0.01 times the group lasso penalty parameters.
            """)
    parser.add_argument('--group-lasso-params',
        type=str,
        default="0.1",
        help="Group lasso penalty parameters, comma separated.")
    parser.add_argument('--ridge-params',
        type=str,
        default='0.0001',
        help="Ridge penalty parameters, comma separated.")
    parser.add_argument('--hidden-sizes',
        type=str,
        default="3",
        help="Comma-separated list of number of hidden nodes")
    parser.add_argument('--n-train',
        type=int,
        default=400,
        help="Number of training examples.")
    parser.add_argument('--n-val',
        type=int,
        default=100,
        help="Number of validation examples.")
    parser.add_argument('--num-p',
        type=int,
        default=50,
        help="Number of features")

    args = parser.parse_args()
    args.ridge_params = process_params(args.ridge_params, float)
    args.lasso_param_ratios = process_params(args.lasso_param_ratios, float)
    args.group_lasso_params = process_params(args.group_lasso_params, float)
    args.hidden_sizes = process_params(args.hidden_sizes, int)

    args.seed_list = process_params(args.seeds, int)
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    args.log_file = make_log_file_name(args)
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    log.info(args)

    pool = Pool(args.num_threads) if args.num_threads > 1 else None
    test_losses = []
    for seed in args.seed_list:
        settings = Settings(
            nn_cls = NeuralNetwork,
            func = six_variable_func,
            n_train = args.n_train,
            n_val = args.n_val,
            n_test = 2000, # Number of test observations
            num_p = args.num_p,
            snr = 2, # Signal to noise ratio
            ridge_param = 0,
            lasso_param = 0,
            group_lasso_param = 0,
            hidden_sizes=[],
            learn_rate = 0.05, # learning rate used in proximal gradient descent
            num_inits = 2,
            data_classes = 0 # 0 for regression, 1 for classification
        )
        best_test_loss = run_simulation(seed, args, settings, pool)
        test_losses.append(best_test_loss)
        log.info("FINAL seed %d, Best Test loss %f" % (seed, best_test_loss))

    log.info("AVG Test loss %s" % get_summary(test_losses))

    if pool is not None:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main(sys.argv[1:])
