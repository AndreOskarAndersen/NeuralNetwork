import numpy as np

class Neuron:
    def __init__(self, input_weights, input_a):
        self.z = 0
        self.a = 0
        self.b = np.random.rand()
        self.grad = 0
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-1 * x))
    
    def sigmoid_derived(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def predict(self, input_weights, input_a):
        self.z = np.matmul(input_weights, input_a) + self.b
        self.a = self.sigmoid(self.z)