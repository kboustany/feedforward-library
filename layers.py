import numpy as np


class Layer:
    """ A fully connected layer in a feedforward neural network. """


    def __init__(self, neurons, activation, dropout=1):

        self.rows = 0
        self.neurons = neurons
        self.activation = activation
        self.dropout = dropout
        self.weight = None
        self.value = None
        self.derivative = None
        self.gradient = None
        self.error = None


    def initialize(self, seed=0):
        """ Initializes the weights of the layer. """

        rng = np.random.default_rng(seed)
        scale = 0

        if self.activation.parameterized:
            scale = np.sqrt(2 / self.rows) # He.

        else:
            scale = np.sqrt(2 / (self.rows + self.neurons)) # Glorot.

        self.weight = rng.normal(scale=scale, size=(self.rows, self.neurons))
        self.gradient = np.zeros(self.weight.shape)


    def feed(self, input, training=False, seed=0):
        """ Performs a forward pass through the layer."""

        if training == True: # During training, a dropout mask is applied.
            
            rng = np.random.default_rng(seed)
            mask = rng.binomial(1, self.dropout, size=input.shape)
            input = input * mask
            self.value = self.activation.value(input @ self.weight) \
                / self.dropout
            self.derivative = self.activation.derivative(input @ self.weight)

        else:
            self.value = self.activation.value(input @ self.weight)
            self.derivative = self.activation.derivative(input @ self.weight)
        

    def clear_values(self):
        """ Clears the stored values after a forward pass. """

        self.value = None
        self.derivative = None


    def clear_gradient(self):
        """ Clears the stored gradient after a backward pass. """

        self.gradient = np.zeros(self.weight.shape)


    def clear(self):
        """ Clears all stored values and gradients after training. """

        self.value = None
        self.derivative = None
        self.gradient = np.zeros(self.weight.shape)
        self.error = None