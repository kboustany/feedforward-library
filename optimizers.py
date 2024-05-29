from utilities import *
from rates import *


class Optimizer:
    """ An optimizer class which implements high-level features of different 
    gradient descent algorithms. """
    
    
    def __init__(self, learning=Constant()):
        
        self.learning = learning


    def _backpropagate(self, layers, x, y, i):
        """ Implements the backpropagation algorithm for feedforward 
        networks. """
         
        layers[-1].error = np.diagflat(layers[-1].derivative[i : i + 1])

        for current, next in zip(layers[: : -1], layers[-2 : : -1]):
            value = next.value[i : i + 1]
            derivative = next.derivative[i : i + 1]
            current.gradient += gradient(value, y, current.error)
            next.error = error(current.error, current.weight, derivative)

        layers[0].gradient += gradient(x, y, layers[0].error)


    def _update(self, layers, size):
        """ Updates the weights of the network during gradient descent. """

        for layer in layers:
            update = (self.learning.rate / size) * layer.gradient
            layer.weight -= update
            layer.clear_gradient()


class BatchDescent(Optimizer):
    """ A full batch gradient descent algorithm. """


    def __init__(self, learning=Constant()):

        super().__init__(learning)

    
    def update(self, network, loss, x, y, epoch):

        p = network.forward(x, training=True, seed=epoch)
        loss.values.append(loss.value(y, p))
        z = loss.derivative(y, p)

        for i in range(x.shape[0]):
            self._backpropagate(network.layers, x[i : i + 1], z[i : i + 1], i)

        self._update(network.layers, x.shape[0], epoch)


class StochasticDescent(Optimizer):
    """ A stochastic (online) gradient descent algorithm. """


    def __init__(self, learning=Constant()):

        super().__init__(learning)

    
    def update(self, network, loss, x, y, epoch):

        for i in range(x.shape[0]):
            p = network.forward(x, training=True, seed=epoch)
            loss.values.append(loss.value(y[i : i + 1], p[i : i + 1])) 
            z = loss.derivative(y[i : i + 1], p[i : i + 1])
            self._backpropagate(network.layers, x[i : i + 1], z, i)
            self._update(network.layers, 1, epoch)


class MiniBatchDescent(Optimizer):
    """ A mini-batch gradient descent algorithm. """
    

    def __init__(self, learning=Constant(), size=32):

        super().__init__(learning)
        self.size = size


    def _batch_update(self, network, loss, x, y, batch, epoch):

        p = network.forward(x, training=True, seed=epoch)
        loss.values.append(loss.value(y, p))
        z = loss.derivative(y, p)

        for i in batch:
            self._backpropagate(network.layers, x[i : i + 1], z[i : i + 1], i)

        self._update(network.layers, len(batch), epoch)
    

    def update(self, network, loss, x, y, epoch):
        """ Performs weight updates one batch at a time and esures the batches 
        are permuted after each training epoch. """

        batches = generate_batches(x.shape[0], self.size)
        rng = np.random.default_rng(epoch)

        while batches != []:
            i = rng.integers(len(batches))
            batch = batches.pop(i)
            self._batch_update(network, loss, x, y, batch, epoch)