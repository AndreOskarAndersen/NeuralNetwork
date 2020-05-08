"""
====================================================================
                        Hidden Layer
--------------------------------------------------------------------
The following is a class representing a hidden layer and the output
layer of the neural network, as well as the input weights of the 
layer. The class takes the amount of dimensions of the layer and the
amount of dimensions of the input layer as parameters.
====================================================================
"""

import numpy as np

class HiddenLayer:
    def __init__(self, dimensions, dim_from):
        self.dimensions = dimensions # Amount of dimensions of the layer
        self.a = np.zeros((dimensions, 1)) # Values of the nodes
        self.b = np.random.uniform(size = (dimensions, 1)) # Bias of each node
        self.grad = np.zeros((dimensions, 1)) # Used for gradient descent
        self.weights = np.random.uniform(size = (dimensions, dim_from)) # Input weights
    
    # Feeds forward the layer
    # 
    def feedforward(self, input_a):
        for i in range(self.dimensions):
            self.a[i] = 1/(1 + np.exp(-1 * (np.matmul(self.weights[i], input_a) + self.b[i])))