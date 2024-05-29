import numpy as np


def rmse(y, p):
    """ Root mean squared error performance metric. """

    return np.sqrt(np.sum(np.square(y - p)) / y.shape[0])

def accuracy(y, p):
    """ Classification accuracy performance metric. """

    x = np.equal(np.argmax(p, axis=1), np.argmax(y, axis=1)).sum()

    return (x / y.shape[0])