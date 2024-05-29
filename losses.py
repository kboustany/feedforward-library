import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Loss:
    """ A loss class which implements high-level features of an arbitrary 
    objective function. """


    def __init__(self):

        self.values = []


    def plot(self):
        """ Plots the evolution of the loss during training."""
        
        sns.lineplot(x=range(len(self.values)), y=self.values)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.grid(True)
        plt.show()
        self.values = []


class MeanSquared(Loss):
    """ A mean squared objective function used for regression tasks. """
    
    
    def __init__(self):

        super().__init__()


    def value(self, y, p):

        return np.sum(np.square(y - p)) / y.shape[0]
    
    
    def derivative(self, y, p):

        return 2 * (p - y)
    
    
class CrossEntropy(Loss):
    """ A cross entropy objective function used for classification tasks. """


    def __init__(self):

        super().__init__()


    def value(self, y, p):

        p = np.clip(p, 1e-9, 1 - 1e-9)

        return -1 * np.sum(y * np.log(p)) / y.shape[0]
    
    
    def derivative(self, y, p):
        
        return p - y