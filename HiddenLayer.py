import numpy as np
from Neuron import Neuron

class HiddenLayer:
    def __init__(self, dimensions, input_weights, input_a):
        self.dimensions = dimensions
        self.input_weights = input_weights
        self.input_a = input_a
        
        self.neurons = []
        for i in range(dimensions):
            self.neurons = np.append(self.neurons, Neuron())
            
        self.neurons = self.neurons.reshape((-1, 1))
        
        self.a = []
        for neuron in self.neurons:
            self.a = np.append(self.a, neuron[0].a)
            
        self.a = self.a.reshape((-1, 1))
        
    def predict(self, input_weights, input_a):
        for i in range(self.dimensions):
            self.neurons[i][0].predict(input_weights.weights[i], input_a)
            
        self.a = []
        for neuron in self.neurons:
            self.a = np.append(self.a, neuron[0].a)
            
        self.a = self.a.reshape((-1, 1))
        
    def predict1(self, input_weights, input_a):
        for i in range(self.dimensions):
            self.neurons[i][0].predict(input_weights.weights[i], input_a)
            
        self.a = []
        for neuron in self.neurons:
            self.a = np.append(self.a, neuron[0].a)
            
        self.a = self.a.reshape((-1, 1))