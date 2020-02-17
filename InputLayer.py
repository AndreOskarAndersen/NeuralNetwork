import numpy as np
from InputNeuron import InputNeuron

class InputLayer:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.a = np.zeros((dimensions, 1))
        
    def feedforward(self, x):
        self.a = x