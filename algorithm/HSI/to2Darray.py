import numpy as np


def to_2d_array(array):
    [m, n, p] = array.shape
    res = np.zeros((p, m*n))
    for i in range(p):
        res[i, :] = np.reshape(array[:, :, i], (1, m*n))
    return res
