from utilities import *
from losses import *


class Trainer:
    """ A trainer class which trains a given neural network with a given 
    optimizer algorithm. """

    def __init__(self, network, optimizer):

        self.network = network
        self.optimizer = optimizer
        if self.network.classification == True:
            self.loss = CrossEntropy()
        else:
            self.loss = MeanSquared()
        self.training_batch = ()
        self.validation_batch = ()
        self.prepared = False
 
    def prepare(self, x, y, size, seed=0):
        """ Initializes the weights of the network and permutes and splits 
        training observations into training and cross-validation sets. """

        self.network.initialize(seed=seed)
        x, y = permute(x, y, seed)
        self.network.x_factors = (np.mean(x[:size]), np.std(x[:size]))
        self.network.y_factors = (np.mean(y[:size]), np.std(y[:size]))
        if self.network.classification == True:
            self.training_batch = \
                add_intercept(normalize(x[:size], self.network.x_factors)), \
                    y[:size]
            self.validation_batch = \
                add_intercept(normalize(x[size:], self.network.x_factors)), \
                    y[size:]
        else:
            self.training_batch = \
                add_intercept(normalize(x[:size], self.network.x_factors)), \
                    normalize(y[:size], self.network.y_factors)
            self.validation_batch = \
                add_intercept(normalize(x[size:], self.network.x_factors)), \
                    normalize(y[size:], self.network.y_factors)
        self.prepared = True

    def fit(self, epochs):
        """ Trains the neural network, implementing a learning rate decay 
        schedule dictated by validation loss. """
        
        self.optimizer.learning.epochs = epochs
        x, y = self.training_batch
        p = self.network.forward(x)
        loss = self.loss.value(y, p)
        for epoch in range(epochs):
            self.optimizer.update(self.network, self.loss, x, y, epoch)
            p = self.network.forward(x)
            if self.loss.value(y, p) >= loss:
                self.optimizer.learning.increment()   
            loss = self.loss.value(y, p)
            x, y = permute(x, y, epoch)
        self.network.trained = True
        print(f'Training loss: {self.loss.values[-1]}.')
        self.loss.plot()
        self.network.clear()
