import numpy as np

CLASSIFICATION_DICT = {
        "six_variable_additive_func": False,
        "six_variable_multivar_func": False,
        "six_variable_mix_func": False,
        "six_variable_nn_func": False,
        "six_variable_additive_binary_func": True,
        "six_variable_multivar_binary_func": True,
        "six_variable_mix_binary_func": True,
}

"""
Regression Function options
"""
def six_variable_multivar_func(xs):
    return np.sin(xs[:,0] * (xs[:,0] + xs[:,1])) * np.cos(xs[:,2] + xs[:,3] * xs[:,4]) * np.sin(np.exp(xs[:,4]) + np.exp(xs[:,5]) - xs[:,1])

def six_variable_mix_func(xs):
    return np.minimum(xs[:,0], xs[:,1]) * np.cos(1.5 * xs[:,2] + 2 * xs[:,3]) + np.exp(xs[:,4] + np.sin(xs[:,3])) * xs[:,1] + np.sin(np.maximum(xs[:,5], xs[:,2])) * (xs[:,4] - xs[:,0])

def six_variable_additive_func(xs):
    return np.sin(2 * xs[:,0]) + np.cos(5 * xs[:,1]) + np.power(xs[:,2], 3) - np.sin(xs[:,3]) + xs[:,4] - np.power(xs[:,5], 2)

def six_variable_nn_func(xs):
    """
    Four hidden nodes
    """
    return np.tanh(xs[:,0] + 2 * xs[:,1] - 3 * xs[:,2] + 2 * xs[:,3]) - 2 * np.tanh(xs[:,0] - xs[:,4] + 2 * xs[:,5]) + np.tanh(-xs[:,1] - xs[:,2] + xs[:,3] - xs[:,5]) + np.tanh(xs[:,4] - 0.5 * xs[:,2] + 0.5 * xs[:,5])


"""
Classification function options
"""
def six_variable_additive_binary_func(xs):
    ys = six_variable_additive_func(xs)
    return 1.0/(1.0 + np.exp(-5.0 * (ys - 0.5)))

def six_variable_multivar_binary_func(xs):
    ys = six_variable_multivar_func(xs)
    return 1.0/(1.0 + np.exp(-4.0 * (ys - 0.1)))

def six_variable_mix_binary_func(xs):
    ys = six_variable_mix_func(xs)
    return 1.0/(1.0 + np.exp(-5.0 * (ys - 1.25)))
