import numpy as np

"""
Example functions to use in regression simulation
"""
def six_variable_func(xs):
    return np.sin(xs[:,0] * (xs[:,0] + xs[:,1])) * np.cos(xs[:,2] + xs[:,3] * xs[:,4]) * np.sin(np.exp(xs[:,4]) + np.exp(xs[:,5]) - xs[:,1])

def six_variable_func_complex(xs):
    return np.minimum(xs[:,0], xs[:,1]) * np.cos(1.5 * xs[:,2] + 2 * xs[:,3]) + np.exp(xs[:,4] + np.sin(xs[:,3])) * xs[:,1] + np.sin(np.maximum(xs[:,5], xs[:,2])) * (xs[:,4] - xs[:,0])

def six_variable_func_additive(xs):
    return np.sin(2 * xs[:,0]) + np.cos(5 * xs[:,1]) + np.power(xs[:,2], 3) - np.sin(xs[:,3]) + xs[:,4] - np.power(xs[:,5], 2)

"""
Example functions to use in classification simulation
"""
def six_variable_additive_binary_func(xs):
    ys = np.sin(2 * xs[:,0]) + np.cos(5 * xs[:,1]) + np.power(xs[:,2], 3) - np.sin(xs[:,3]) + xs[:,4] - np.power(xs[:,5], 2)
    return 1.0/(1.0 + np.exp(-5.0 * (ys - 0.5)))

def six_variable_binary_func(xs):
    ys = np.sin(xs[:,0] * (xs[:,0] + xs[:,1])) * np.cos(xs[:,2] + xs[:,3] * xs[:,4]) * np.sin(np.exp(xs[:,4]) + np.exp(xs[:,5]) - xs[:,1])
    return 1.0/(1.0 + np.exp(-4.0 * (ys - 0.1)))

def six_variable_binary_func_complex(xs):
    ys = np.minimum(xs[:,0], xs[:,1]) * np.cos(1.5 * xs[:,2] + 2 * xs[:,3]) + np.exp(xs[:,4] + np.sin(xs[:,3])) * xs[:,1] + np.sin(np.maximum(xs[:,5], xs[:,2])) * (xs[:,4] - xs[:,0])
    return 1.0/(1.0 + np.exp(-5.0 * (ys - 1.25)))

class Dataset:
    def __init__(self, x_train=None, y_train=None, y_train_true=None, x_val=None, y_val=None, y_val_true=None, x_test=None, y_test=None, y_test_true=None):
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_true = y_train_true
        self.x_val = x_val
        self.y_val = y_val
        self.y_val_true = y_val_true
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_true = y_test_true

class DataGenerator:
    def __init__(self, settings):
        self.n_train = settings.n_train
        self.n_val = settings.n_val
        self.n_test = settings.n_test
        self.num_p = settings.num_p
	self.data_classes = settings.data_classes
        self.func = settings.func
        self.snr = settings.snr

    def create_data(self):
        x_train, y_train, y_train_true = self._create_data(self.n_train)
        x_val, y_val, y_val_true = self._create_data(self.n_val)
        x_test, y_test, y_test_true = self._create_data(self.n_test)
        return Dataset(
            x_train, y_train, y_train_true,
            x_val, y_val, y_val_true,
            x_test, y_test, y_test_true,
        )

    def _create_data(self, size_n):
        if size_n <= 0:
            return None, None, None

        xs = np.random.rand(size_n, self.num_p)
	if self.data_classes == 0:
	    # regression
            true_ys = self.func(xs)
            true_ys = np.reshape(true_ys, (true_ys.size, 1))
            eps = np.random.randn(size_n, 1)
            eps_norm = np.linalg.norm(eps)
            y_norm = np.linalg.norm(true_ys)
            y = true_ys + 1. / self.snr * y_norm / eps_norm * eps
	elif self.data_classes == 1:
	    # classification
            true_ys = self.func(xs)
            true_ys = np.reshape(true_ys, (true_ys.size, 1))
	    y = np.array(np.random.random_sample((true_ys.size, 1)) < true_ys, dtype=int)
        return xs, y, true_ys
