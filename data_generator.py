import numpy as np

class Dataset:
    def __init__(self, x, y, y_true=None):
        self.x = x
        self.y = y
        self.y_true = y_true

    def create_restricted(self, max_relevant_idx):
        return Dataset(
                self.x[:, :max_relevant_idx],
                self.y,
                self.y_true)

class DataGenerator:
    def __init__(self, num_p, func, is_classification=False, snr=0):
        self.num_p = num_p
        self.func = func
        print(is_classification)
        self.is_classification = is_classification
        self.snr = snr

    def create_data(self, n_obs):
        assert n_obs > 0

        xs = np.random.rand(n_obs, self.num_p)
        if not self.is_classification:
	    # regression
            true_ys = self.func(xs)
            true_ys = np.reshape(true_ys, (true_ys.size, 1))
            eps = np.random.randn(n_obs, 1)
            eps_norm = np.linalg.norm(eps)
            y_norm = np.linalg.norm(true_ys)
            y = true_ys + 1. / self.snr * y_norm / eps_norm * eps
        else:
	    # classification
            true_ys = self.func(xs)
            true_ys = np.reshape(true_ys, (true_ys.size, 1))
            y = np.array(np.random.random_sample((true_ys.size, 1)) < true_ys, dtype=int)
        return Dataset(xs, y, true_ys)
