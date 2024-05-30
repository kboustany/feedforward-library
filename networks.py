import numpy as np
from utilities import *


class Network:
    """ A feedforward neural network which can perform either regression or 
    classification tasks, built one layer at a time. """

    def __init__(self, covariates, layers, classification=False):

        self.layers = layers
        self.layers[0].rows = covariates + 1
        for current, next in zip(layers, layers[1 : ]):
            next.rows = current.neurons
        self.classification = classification
        self.initialized = False
        self.trained = False
        self.x_factors = (0, 1)
        self.y_factors = (0, 1)

    def initialize(self, seed=0):
        """ Initializes the layers of the network. """

        for layer in self.layers:
            layer.initialize(seed)
        self.initialized = True

    def forward(self, x, training=False, seed=0):
        """ Performs a forward pass during training and cross-validation. """
    
        x = convert(x)
        for layer in self.layers:
            layer.feed(x, training=training, seed=seed)
            x = layer.value
        if self.classification == True:
            x = softmax(x)
            
        return convert(x)

    def predict(self, x):
        """ Outputs a prediction of the network after training. """

        x = convert(x)
        x = normalize(x, self.x_factors)
        x = add_intercept(x)
        for layer in self.layers:
            layer.feed(x)
            x = layer.value
            layer.clear_values()
        if self.classification == True:
            x = softmax(x)
        else:
            x = unormalize(x, self.y_factors)

        return convert(x)

    def clear(self):
        """ Clears the stored values and gradients from the layers of the 
        network after training. """
        
        for layer in self.layers:
            layer.clear()
