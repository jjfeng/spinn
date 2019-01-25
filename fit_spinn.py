import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import logging
from scipy.stats import pearsonr

from common import *
from neural_network import NeuralNetwork
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='seed',
        default=1)
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
        default="_output/log.txt")
    parser.add_argument('--out-file',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--summary-file',
        type=str,
        default="_output/fitted_spinn.csv")
    parser.add_argument('--is-relu',
        type=int,
        default=0)
    parser.add_argument('--lasso-param-ratios',
        type=str,
        default="0.1,0.001")
    parser.add_argument('--group-lasso-params',
        type=str,
        default="1")
    parser.add_argument('--ridge-params',
        type=str,
        default='0.0001')
    parser.add_argument('--hidden-size-options',
        type=str,
        default='2:2,5:2')
    parser.add_argument('--data-classes',
        type=int,
        default=0)
    parser.add_argument('--learn-rate',
        type=float,
        default=0.05)
    parser.add_argument('--kfold',
        type=int,
        default=2)
    parser.add_argument('--max-iters',
        type=int,
        default=10)
    parser.add_argument('--num-inits',
        type=int,
        default=1)
    parser.add_argument('--num-jobs',
        type=int,
        help='number of parallel jobss',
        default=1)
    parser.add_argument('--bayes-opt',
        action='store_true')
    parser.add_argument('--num-bayes-iters',
        type=int,
        default=1)
    parser.add_argument('--num-bayes-inits',
        type=int,
        default=1)

    args = parser.parse_args()
    args.ridge_params = process_params(args.ridge_params, float)
    args.lasso_param_ratios = process_params(args.lasso_param_ratios, float)
    args.group_lasso_params = process_params(args.group_lasso_params, float)
    args.hidden_size_options = [
            list(map(int, hidden_size_str.split(":"))) if hidden_size_str != "0" else []
            for hidden_size_str in args.hidden_size_options.split(",")]
    return args

def fit_spinn(args, train_data, param_grid, param_limits):
    if args.bayes_opt:
        best_params, cv_results = _get_best_params_bayesopt(
            param_grid["layer_sizes"],
            param_limits,
            train_data,
            args.kfold,
            args.num_bayes_inits,
            args.num_bayes_iters,
            args.num_jobs)
    else:
        best_params, cv_results = _get_best_params_gridsearch(
                param_grid,
                train_data,
                args.kfold,
                args.num_jobs)
    logging.info("Best params %s", str(best_params))
    best_params["num_inits"] = args.num_inits

    # Fit for the full conditional mean
    final_nn = NeuralNetwork(**best_params)
    final_nn.fit(train_data.x, train_data.y)
    return final_nn, cv_results

def _get_best_params_bayesopt(
        layer_size_options,
        param_limits,
        dataset,
        cv=2,
        num_bayes_inits=1,
        num_bayes_iters=1,
        num_jobs=2):
    """
    Runs cross-validation if needed
    @return best params chosen by bayesian optimization
    """
    def _fit_spinn_params(
                layer_size_idx,
                data_classes,
                lasso_param_log10_ratio,
                group_lasso_log10_param,
                ridge_log10_param,
                max_iters,
                num_inits,
                init_learn_rate,
                is_relu):
        # This is a grid with only one option
        # We do this since GridsearchCV is a nice wrapper for the CV we want to do
        param_grid = {
                "layer_sizes": layer_size_options[int(layer_size_idx)],
                "data_classes": int(data_classes),
                "lasso_param_ratio": np.power(10.0, lasso_param_log10_ratio) if np.isfinite(lasso_param_log10_ratio) else 0,
                "group_lasso_param": np.power(10.0, group_lasso_log10_param) if np.isfinite(group_lasso_log10_param) else 0,
                "ridge_param": np.power(10.0, ridge_log10_param),
                "max_iters": int(max_iters),
                "num_inits": int(num_inits),
                "init_learn_rate": init_learn_rate,
                "is_relu": is_relu}
        param_grid = {k: [v] for k, v in param_grid.items()}
        grid_search_cv = GridSearchCV(
                NeuralNetwork(),
                param_grid=param_grid,
                cv=cv,
                n_jobs=num_jobs,
                refit=False)
        grid_search_cv.fit(dataset.x, dataset.y)
        logging.info("Completed CV")
        logging.info(grid_search_cv.cv_results_)
        return grid_search_cv.best_score_

    only_one_option = np.all([v[0] == v[1] for k,v in param_limits.items()])
    if len(layer_size_options) == 1 and only_one_option:
        # Don't do anything if there is nothing to tune
        best_params = {k:v[0] for k, v in param_limits.items()}
        best_params["layer_size_idx"] = 0
        all_results = None
    elif only_one_option:
        # Don't do any Bayesian opt if we only want to compare layer size options
        all_results = {"params": [], "values": []}
        for layer_idx, layer_sizes in enumerate(layer_size_options):
            param_dict = {k:v[0] for k,v in param_limits.items()}
            param_dict["layer_size_idx"] = layer_idx
            score = _fit_spinn_params(**param_dict)
            all_results["params"].append(param_dict)
            all_results["values"].append(score)
        best_idx = np.argmax([score for score in all_results["values"]])
        best_params = all_results["params"][best_idx]
    else:
        # Do bayesian opt for each layer size option
        #gp_params = {"alpha": 1e-5}

        param_limits["layer_size_idx"] = (0, len(layer_size_options) - 0.99)
        nnBO = BayesianOptimization(
                _fit_spinn_params,
                param_limits)
        nnBO.maximize(
                init_points=num_bayes_inits,
                n_iter=num_bayes_iters)
        all_results = nnBO.res["all"]
        best_params = nnBO.res["max"]["max_params"]

    best_params["layer_sizes"] = layer_size_options[int(best_params["layer_size_idx"])]
    best_params["lasso_param_ratio"] = np.power(
            10.0,
            best_params["lasso_param_log10_ratio"]) if np.isfinite(best_params["lasso_param_log10_ratio"]) else 0
    best_params["group_lasso_param"] = np.power(
            10.0,
            best_params["group_lasso_log10_param"]) if np.isfinite(best_params["group_lasso_log10_param"]) else 0
    best_params["ridge_param"] = np.power(
            10.0,
            best_params["ridge_log10_param"])
    best_params.pop("layer_size_idx", None)
    best_params.pop("lasso_param_log10_ratio", None)
    best_params.pop("group_lasso_log10_param", None)
    best_params.pop("ridge_log10_param", None)

    logging.info("Best CV score %f", np.max(all_results["values"]))

    return best_params, all_results


def _get_best_params_gridsearch(param_grid, dataset, cv = 2, num_jobs = 1):
    """
    Runs cross-validation if needed
    @return best params chosen by CV in dict form, `cv_results_` attr from GridSearchCV
    """
    if np.all([len(v) == 1 for k,v in param_grid.items()]):
        # Don't run CV if there is nothing to tune
        return {k:v[0] for k,v in param_grid.items()}, None
    else:
        # grid search CV to get argmins
        grid_search_cv = GridSearchCV(
                NeuralNetwork(),
                param_grid = param_grid,
                cv = cv,
                n_jobs=num_jobs,
                refit=False)

        ### do cross validation
        grid_search_cv.fit(dataset.x, dataset.y)
        logging.info("Completed CV")
        logging.info(grid_search_cv.cv_results_)
        logging.info("Best CV score %f", np.max(grid_search_cv.cv_results_["mean_test_score"]))

        return grid_search_cv.best_params_, grid_search_cv.cv_results_


def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(args)
    # Read data
    train_data, test_data = read_input_data(args)

    num_p = train_data.x.shape[1]
    print("num p", num_p)
    print(train_data.y.mean())

    param_grid = {
        'layer_sizes': [[num_p] + h for h in args.hidden_size_options],
        'data_classes': [args.data_classes],
        'lasso_param_ratio': args.lasso_param_ratios,
        'group_lasso_param': args.group_lasso_params,
        'ridge_param': args.ridge_params,
        'max_iters': [args.max_iters],
        'num_inits': [1],
        'init_learn_rate': [args.learn_rate],
        'is_relu': [args.is_relu],
    }
    param_limits = {
        'data_classes': (args.data_classes, args.data_classes),
        'lasso_param_log10_ratio': (
            np.log10(np.min(args.lasso_param_ratios)),
            np.log10(np.max(args.lasso_param_ratios))),
        'group_lasso_log10_param': (
            np.log10(np.min(args.group_lasso_params)),
            np.log10(np.max(args.group_lasso_params))),
        'ridge_log10_param': (
            np.log10(np.min(args.ridge_params)),
            np.log10(np.max(args.ridge_params))),
        'max_iters': (args.max_iters, args.max_iters),
        'num_inits': (1, 1),
        'init_learn_rate': (args.learn_rate, args.learn_rate),
        'is_relu': (args.is_relu, args.is_relu),
    }
    print("PARAM LIMITS", param_limits)

    final_nn, cv_results = fit_spinn(args, train_data, param_grid, param_limits)

    with open(args.out_file, "wb") as f:
        pickle.dump({
            "model_params": final_nn.model_params,
            "all_results": cv_results}, f)

    # Test loss using the TRUE value of y
    y_pred = final_nn.predict(test_data.x)
    print("nonzero first layer", final_nn.model_params.nonzero_first_layer)
    if args.data_classes == 0:
        test_loss = get_regress_loss(y_pred, test_data.y_true)
        logging.info("pearson %s", pearsonr(y_pred.ravel(), test_data.y_true.ravel()))
        logistic_loss = 0
    else:
        test_loss = 1 - get_classification_accuracy(y_pred, test_data.y_true)
        logistic_loss = get_logistic_loss(y_pred, test_data.y_true)

    num_nonzero_inputs = final_nn.model_params.nonzero_first_layer[0].size
    result = pd.DataFrame({
        "test_loss": [test_loss],
        "logistic_loss": [logistic_loss],
        "num_nonzero": [num_nonzero_inputs]})
    result.T.to_csv(args.summary_file)
    logging.info("DONE!")

if __name__ == "__main__":
    main(sys.argv[1:])
