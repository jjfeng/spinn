import sys
import os
import random
import argparse
import logging
import pickle
import subprocess
import pandas as pd
from scipy.stats import pearsonr

import numpy as np
from common import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=1)
    parser.add_argument('--kfold',
        type=int,
        default=3)
    parser.add_argument('--data-index-file',
        type=str,
        default=None)
    parser.add_argument('--data-X-file',
        type=str)
    parser.add_argument('--data-y-file',
        type=str)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/lasso.log")
    parser.add_argument('--out-file',
        type=str,
        default="_output/out.csv")
    parser.add_argument('--data-classes',
        type=int,
        default=0)
    parser.add_argument('--scratch',
        type=str,
        default="_output/scratch")

    args = parser.parse_args()
    return args

def fit_lasso(
        args,
        train_file,
        test_file):
    cmd = [
        'Rscript',
        '../R/fit_lasso.R',
        train_file,
        test_file,
        args.out_file,
        args.data_classes,
        args.seed,
        args.kfold,
    ]
    cmd = list(map(str, cmd))

    print("Calling:", " ".join(cmd))
    res = subprocess.call(cmd)
    # Check that process complete successfully
    assert res == 0

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)

    #randnum = random.randint(1, 100000)
    #train_file_name = os.path.join(args.scratch, "train%d" % randnum)
    #test_file_name = os.path.join(args.scratch, "test%d" % randnum)
    #write_data_pkl_to_csv(
    #        None,
    #        train_file_name,
    #        test_file_name)
    #fit_lasso(args, train_file_name, test_file_name)
#
#    os.remove(train_file_name)
#    os.remove(test_file_name)

    train_data, test_data = read_input_data(args)

    if args.data_classes == 0:
        lasso = Lasso(
                random_state=args.seed, max_iter=10000)
        alphas = np.power(10., np.arange(2, -4, -0.1))
        tuned_parameters = [{'alpha': alphas}]
    elif args.data_classes == 1:
        lasso = LogisticRegression(
                random_state=args.seed,
                max_iter=10000,
                penalty='l1')
        Cs = np.power(10., np.arange(4, -3, -0.1))
        tuned_parameters = [{'C': Cs}]
    else:
        raise ValueError("huh?")

    clf = GridSearchCV(
            lasso,
            tuned_parameters,
            cv=args.kfold,
            refit=True,
            n_jobs=10)
    clf.fit(train_data.x, train_data.y.ravel())
    logging.info(clf.cv_results_["mean_test_score"])
    logging.info(clf.cv_results_["params"])
    logging.info(clf.best_params_)
    y_pred = clf.predict(test_data.x)
    num_nonzero_inputs = np.sum(np.abs(clf.best_estimator_.coef_) > THRES)
    if args.data_classes == 0:
        test_loss = get_regress_loss(y_pred, test_data.y_true)
        logging.info("pearsonr %s", pearsonr(y_pred.ravel(), test_data.y_true.ravel()))
        logistic_loss = 0
    else:
        test_loss = 1 - get_classification_accuracy(y_pred, test_data.y_true)
        logistic_loss = get_logistic_loss(y_pred, test_data.y_true)

    result = pd.DataFrame({
        "test_loss": [test_loss],
        "logistic_loss": [logistic_loss],
        "num_nonzero": [num_nonzero_inputs]})
    result.T.to_csv(args.out_file)
    logging.info("DONE!")

if __name__ == "__main__":
    main(sys.argv[1:])

