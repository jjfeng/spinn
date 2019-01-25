import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import logging
from scipy.stats import pearsonr

from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='seed',
        default=1)
    parser.add_argument('--max-depth',
        type=int,
        default=10)
    parser.add_argument('--num-trees',
        type=int,
        default=1000)
    parser.add_argument('--num-jobs',
        type=int,
        default=10)
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
        default="_output/log_trees.txt")
    parser.add_argument('--out-file',
        type=str,
        default="_output/fitted_trees.csv")
    parser.add_argument('--data-classes',
        type=int,
        default=0)
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)
    logging.info(args)

    train_data, test_data = read_input_data(args)

    if args.data_classes == 0:
        regr = RandomForestRegressor(
                max_depth=args.max_depth,
                random_state=args.seed,
                n_estimators=args.num_trees,
                max_features=0.3,
                n_jobs=args.num_jobs)
    else:
        train_data.y = train_data.y.astype(int)
        train_data.y_true = train_data.y_true.astype(int)
        test_data.y = test_data.y.astype(int)
        test_data.y_true = test_data.y_true.astype(int)
        regr = RandomForestClassifier(
                max_depth=args.max_depth,
                random_state=args.seed,
                n_estimators=args.num_trees,
                max_features=0.3,
                n_jobs=args.num_jobs)
    regr.fit(train_data.x, train_data.y.ravel())
    logging.info("FEATURE IMPORT")
    sort_idxs = np.argsort(regr.feature_importances_)
    for idx in sort_idxs[-50:]:
        importance = regr.feature_importances_[idx]
        logging.info("%d: %f", idx, importance)

    y_pred = regr.predict(test_data.x)
    if args.data_classes == 0:
        test_loss = get_regress_loss(y_pred, test_data.y_true)
        logging.info("pearsonr %s", pearsonr(y_pred.ravel(), test_data.y_true.ravel()))
        print("PRED", y_pred)
        print(test_data.y_true)
    else:
        test_loss = 1 - get_classification_accuracy(y_pred, test_data.y_true)
    result = pd.DataFrame({
        "test_loss": [test_loss]})
    result.T.to_csv(args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
