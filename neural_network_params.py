import numpy as np
from common import THRES

class NeuralNetworkParams:
    def __init__(self, coefs, intercepts, scaler):
        self.coefs = coefs
        self.intercepts = intercepts
        self.scaler = scaler

    @property
    def nonzero_first_layer(self):
        return np.where(np.max(np.abs(self.coefs[0]), axis=1) > THRES)
