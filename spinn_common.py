import numpy as np
import time
import argparse
import cPickle as pickle
import tensorflow as tf
import numpy as np
import logging as log

from data_generator import DataGenerator
from settings import *
from common import *
from neural_network import NeuralNetwork
from neural_network_wrapper import NeuralNetworkWrapper, NeuralNetworkWrapperKFold

def run_simulation(seed, args, settings, pool, dataset=None, kfold=0, shuffled_idx=None, final_num_inits=10):
    log.info("========= SEED %d =========" % seed)
    pkl_file = make_model_file_name(args, seed)
    log.info("pkl_file %s" % pkl_file)

    st_time = time.time()
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset is None:
        dataset = DataGenerator(settings).create_data()

    data_file = make_data_file_name(args, seed)
    log.info("data_file %s" % data_file)
    if data_file is not None:
        if shuffled_idx is None:
            with open(data_file, "w") as f:
                pickle.dump(dataset, f, protocol=-1)
        else:
            with open(data_file.replace("pkl", "csv"), "w") as f:
                f.writelines(["%d\n" % i for i in shuffled_idx.tolist()])

    best_nn_hist = []
    best_nn_wrap = None
    best_settings = None
    best_val_loss = 0
    start_seed = np.random.randint(100000)
    for h in args.hidden_sizes:
        for r in args.ridge_params:
            for gl in args.group_lasso_params:
                for l_ratio in args.lasso_param_ratios:
                    if gl != 0:
                        l = l_ratio * gl
                    else:
                        l = l_ratio
                    settings_mini = SettingsMini([h], r, lasso_param=l, group_lasso_param=gl)
                    settings.update(settings_mini)

                    if kfold > 0:
                        nn_wrap = NeuralNetworkWrapperKFold(settings)
                        val_loss = nn_wrap.fit(dataset, pool, start_seed=start_seed, kfold=kfold)
                    else:
                        nn_wrap = NeuralNetworkWrapper(settings)
                        nn_wrap.fit(dataset, pool, start_seed=start_seed)
                        # Assess validation loss - we pick the neural network parameters that minimize the val loss
                        y_val_pred = nn_wrap.predict(dataset.x_val, dataset.y_val)
                        if settings.data_classes == 0:
                            val_loss = get_regress_loss(y_val_pred, dataset.y_val)
                        else:
                            val_loss = 1 - get_classification_accuracy(y_val_pred, dataset.y_val)

                    start_seed += 1
                    log.info("Params %s: validation loss %f" % (settings_mini, val_loss))
                    log.info("Time %s" % str(time.time() - st_time))
                    if best_nn_wrap is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_nn_wrap = nn_wrap
                        best_settings = settings_mini

                        # Save our best model so far
                        log.info("Seed %d, Best Params %s: validation loss %f" % (seed, best_settings, best_val_loss))
                        if kfold == 0:
                            y_pred = best_nn_wrap.predict(dataset.x_test, dataset.y_test)
                            # Test loss using the TRUE value of y
                            if settings.data_classes == 0:
                                test_loss = get_regress_loss(y_pred, dataset.y_test_true)
                            else:
                                test_loss = 1 - get_classification_accuracy(y_pred, dataset.y_test_true)
                            log.info("Seed %d, Test loss %f" % (seed, test_loss))
                            best_nn_wrap.best_nn_res.test_loss = test_loss

                            best_nn_hist.append(
                                (best_nn_wrap.best_nn_res, test_loss)
                            )
                        else:
                            best_nn_hist.append(
                                (best_nn_wrap.best_nn_res, None)
                            )
                        with open(pkl_file, "w") as f:
                            pickle.dump(best_nn_hist, f, protocol=-1)

    if kfold > 0:
        log.info("Doing final retraining!")
        # We need to retrain the model now
        settings.update(best_settings)
        settings.num_inits = final_num_inits
        best_nn_wrap = NeuralNetworkWrapper(settings)
        best_nn_wrap.fit(dataset, pool, start_seed=start_seed)

        # Test loss using the TRUE value of y
        y_pred = best_nn_wrap.predict(dataset.x_test, dataset.y_test)
        if settings.data_classes == 0:
            test_loss = get_regress_loss(y_pred, dataset.y_test_true)
        else:
            test_loss = 1 - get_classification_accuracy(y_pred, dataset.y_test_true)
        log.info("Seed %d, Test loss %f" % (seed, test_loss))
        best_nn_wrap.best_nn_res.test_loss = test_loss

        best_nn_hist.append(best_nn_wrap.best_nn_res)
        with open(pkl_file, "w") as f:
            pickle.dump(best_nn_hist, f, protocol=-1)

    return best_nn_wrap.best_nn_res.test_loss
