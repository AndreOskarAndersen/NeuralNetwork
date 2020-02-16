import numpy as np
from Neuron import Neuron

""" Class for a hidden layer and the output layer """
class HiddenLayer:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        
        self.neurons = []
        for _ in range(dimensions):
            self.neurons = np.append(self.neurons, Neuron())
            
        self.neurons = self.neurons.reshape((-1, 1))
        
        self.a = [] # Collection of the values of all the neurons in the layer
        for neuron in self.neurons:
            self.a = np.append(self.a, neuron[0].a)
            
        self.a = self.a.reshape((-1, 1))
        
    def feedforward(self, input_weights, input_a):
        for i in range(self.dimensions):
            self.neurons[i][0].feedforward(input_weights.weights[i], input_a)
        
        self.a = [] # Collection of the values of all the neurons in the layer
        for neuron in self.neurons:
            self.a = np.append(self.a, neuron[0].a)
            
        self.a = self.a.reshape((-1, 1))
        
    def predict(self, input_weights, input_a):
        self.feedforward(input_weights, input_a)