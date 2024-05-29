import numpy as np


class Linear:
    """ A linear (identity) activation and its derivative. """

    
    def __init__(self):

        self.parameterized = False


    def value(self, x):

        return x
    
    
    def derivative(self, x):

        return np.ones(x.shape)
    
    
class ReLU:
    """ A (leaky) ReLU activation and its derivative. """


    def __init__(self, alpha=0):

        self.parameterized = True
        self.alpha = alpha


    def value(self, x):

        return np.maximum(self.alpha * x, x)
    
    
    def derivative(self, x):

        return np.sign(np.maximum(0, x)) + \
            (self.alpha * np.sign(np.maximum(0, -x)))
    
    
class Sigmoid:
    """ A sigmoid activation and its derivative. """


    def __init__(self):

        self.parameterized = False


    def value(self, x):

        return 1 / (1 + np.exp(-x))
    
    
    def derivative(self, x):

        return self.value(x) * (1 - self.value(x))
    
    
class Tanh:
    """ A hyperbolic tangent activation and its derivative. """


    def __init__(self):

        self.parameterized = False


    def value(self, x):

        return np.tanh(x)
    
    
    def derivative(self, x):
        
        return 1 - np.square(self.value(x))