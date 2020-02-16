import numpy as np

class Neuron:
    def __init__(self):
        self.a = 0 # Value of neuron after activation function applied
        self.b = np.random.rand() # Bias
        self.grad = 0 # Gradient
    
    def feedforward(self, input_weights, input_a):
        self.a = 1/(1 + np.exp(-1 * (np.matmul(input_weights, input_a) + self.b)))