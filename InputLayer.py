import numpy as np
from InputNeuron import InputNeuron

class InputLayer:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        
        self.neurons = []
        for _ in range(dimensions):
            self.neurons = np.append(self.neurons, InputNeuron())
            
        self.neurons = self.neurons.reshape((-1, 1))
        
        self.a = []
        for neuron in self.neurons:
            self.a = np.append(self.a, neuron[0].a)
            
        self.a = self.a.reshape((-1, 1))
        
    def feedforward(self, x):
        self.a = np.array(x).reshape((-1, 1))
        
        for i in range(self.dimensions):
            self.neurons[i][0].a = x[i]
        
    def predict(self, x):
        self.a = np.array(x).reshape((-1, 1))
        
        for i in range(self.dimensions):
            self.neurons[i][0].a = x[i]