import numpy as np


def softmax(x):
    """ Applies the softmax function to an array. """
    
    x = np.exp(x - np.max(x))
    sum = np.sum(x, axis=1).reshape((x.shape[0], 1))

    return np.reciprocal(sum) * x
    

def permute(x, y, seed=0):
    """ Permutes the rows of a pair of arrays along the same permutation. """

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])

    return x[perm], y[perm]
    

def normalize(x, a):
    """ Normalizes the elements of an array to a given mean and variance. """

    return (x - a[0]) / a[1]
    

def unormalize(x, a):
    """ Reverses the action of the normalize function. """

    return (a[1] * x) + a[0]
    

def add_intercept(x):
    """ Adds a columns of ones to an array to accoutnt for bias terms. """

    return np.hstack((np.ones((x.shape[0], 1)), x))
    

def convert(x):
    """ Converts a scalar to an array of shape (1, 1), and vice versa. """

    if isinstance(x, (int, float)):

        return np.reshape(x, newshape=(1, 1))
    
    elif x.shape == (1, 1):

        return x.item()
    
    else:

        return x   


def error(x, y, z):
    """ Computes the error term during backpropagation. """

    return (x @ np.transpose(y)) * z


def gradient(x, y, z):
    """ Computes the gradient over the entries of a single observation. """

    gradient = np.zeros((x.shape[1], z.shape[1]))
    for i in range(y.shape[1]):
        gradient += y[ : , i] * (np.transpose(x) @ z[i : i + 1])

    return gradient


def generate_batches(n, d):
    """ Generates batches of integers for use in mini-batch descent. """

    batches = []
    i = 0
    while (i + 1) * d < n:
        batches.append(range(i * d, (i + 1) * d))
        i += 1
    batches.append(range(i * d, n))
    
    return batches


def train_test_split(x, y, size):
    """ Splits data into training and testing sets. """
    
    x_train = x[:size]
    y_train = y[:size]
    x_test = x[size:]
    y_test = y[size:]

    return x_train, y_train, x_test, y_test
