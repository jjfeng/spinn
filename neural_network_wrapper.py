import logging as log
from parallel_worker import ParallelWorker
from parallel_worker import MultiprocessingManager

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from common import THRES, get_classification_accuracy
from data_generator import Dataset

ALMOST_ZERO = 1e-8

class NeuralNetworkWrapperKFold:
    def __init__(self, settings):
        self.settings = settings

    def fit(self, dataset, pool, print_iter=100, start_seed=1, kfold=5):
        n_train = dataset.y_train.size
        fold_indices = np.arange(0, n_train, step=int(n_train/kfold)+ 1, dtype=int)
        fold_indices = np.concatenate([fold_indices, [n_train]])
        kfold_datasets = []
        for k in range(kfold):
            train_idx = np.concatenate([
                np.arange(start=0, stop=fold_indices[k]),
                np.arange(start=fold_indices[k + 1], stop=n_train)
            ])
            val_idx = np.arange(fold_indices[k], fold_indices[k + 1])
            fold_dataset = Dataset(
                x_train=dataset.x_train[train_idx,:],
                y_train=dataset.y_train[train_idx,:],
                x_val=dataset.x_train[val_idx,:],
                y_val=dataset.y_train[val_idx,:],
                x_test=dataset.x_test,
                y_test=dataset.y_test,
            )
            kfold_datasets.append(fold_dataset)

        nn_workers = [
            NeuralNetworkWorker(start_seed + i, self.settings, kfold_dataset, print_iter)
            for i in range(self.settings.num_inits) for kfold_dataset in kfold_datasets
        ]

        # Run neural network training for different initializations in parallel
        if pool is not None:
            mgr = MultiprocessingManager(pool, nn_workers)
            nn_res = mgr.run()
        else:
            nn_res = [nn_w.run() for nn_w in nn_workers]

        nn_res_kfold = []
        self.best_nn_res = []
        for k in range(kfold):
            offset = k * self.settings.num_inits
            train_loss_k = [
                nn_res[offset + i].train_loss
                for i in range(self.settings.num_inits)
            ]
            best_train_k = np.argmin(train_loss_k)
            best_nn_k = nn_res[offset + best_train_k]
            self.best_nn_res.append(best_nn_k)
            val_loss_k = best_nn_k.val_loss
            nn_res_kfold.append(val_loss_k)
        log.info("Completed kfol %s" % nn_res_kfold)
        return np.mean(nn_res_kfold)


class NeuralNetworkWrapper:
    def __init__(self, settings):
        self.settings = settings

    def fit(self, dataset, pool, print_iter=100, start_seed=1):
        nn_workers = [
            NeuralNetworkWorker(start_seed + i, self.settings, dataset, print_iter)
            for i in range(self.settings.num_inits)
        ]

        # Run neural network training for different initializations in parallel
        if pool is not None:
            mgr = MultiprocessingManager(pool, nn_workers)
            nn_res = mgr.run()
        else:
            nn_res = [nn_w.run() for nn_w in nn_workers]

        # Find the best neural network
        train_losses = [nn.train_loss for nn in nn_res]
        log.info("Train losses %s" % train_losses)
        argmin = np.argmin(train_losses)

        # Since this is not kfold CV, recreate best NN
        self.best_nn_res = nn_res[argmin]
        self.best_nn = self.settings.nn_cls(self.settings)
        self.best_nn.model_params = self.best_nn_res.params
        self.best_nn.fit_scaler(dataset)

    def predict(self, x, y):
        return self.best_nn.predict(x, y)

class NeuralNetworkResult:
    def __init__(self, params, scaler, train_loss, val_loss):
        self.params = params
        self.scaler = scaler
        self.train_loss = train_loss
        self.val_loss = val_loss

class NeuralNetworkParams:
    def __init__(self, coefs, intercepts):
        self.nonzero_first_layer = np.where(np.max(np.abs(coefs[0]), axis=1) > THRES)
        self.coefs = coefs
        self.intercepts = intercepts

class NeuralNetworkWorker(ParallelWorker):
    def __init__(self, seed, settings, dataset, print_iter):
        """
        @param seed: a seed for for each parallel worker
        @param nn: NeuralNetwork
        """
        self.seed = seed
        self.settings = settings
        self.dataset = dataset
        self.print_iter = print_iter

    def run_worker(self):
        """
        @return NeuralNetworkResult
        """
        nn = self.settings.nn_cls(self.settings)
        nn.fit(self.dataset, print_iter=self.print_iter)
        return NeuralNetworkResult(nn.model_params, nn.scaler, nn.all_pen_train_err, nn.val_loss)

    def __str__(self):
        """
        @return: string for identifying this worker in an error
        """
        return "Neural Network Worker %d" % self.seed
