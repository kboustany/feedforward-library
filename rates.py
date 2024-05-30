class LearningRate:
    """ A learning rate class which implements high-level features of different 
    learning rate decay schedules. """

    def __init__(self, initial=0.01, final=0.0001):

        self.initial = initial
        self.final = final
        self.rate = self.initial
        self.step = 0
        self.epochs = 1
        

class Constant(LearningRate):
    """ A constant (no decay) learning rate schedule. """

    def __init__(self, initial=0.01, final=0.0001):

        super().__init__(initial, final)

    def increment(self):
         
        self.step += 1
        self.rate = self.initial


class Convex(LearningRate):
    """ A convex (linear) learning rate decay schedule. """

    def __init__(self, initial=0.01, final=0.0001):

        super().__init__(initial, final)

    def increment(self):
         
        self.step += 1
        self.rate = self.initial - (self.initial - self.final) * \
            (self.step / self.epochs)


class Exponential(LearningRate):
    """ An exponential learning rate decay schedule. """
        
    def __init__(self, initial=0.01, final=0.0001):
             
         super().__init__(initial, final)

    def increment(self):
             
        self.step += 1
        self.rate = self.initial * \
            ((self.final / self.initial)**(self.step / self.epochs))
