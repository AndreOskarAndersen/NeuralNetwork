import numpy as np
from HiddenLayer import HiddenLayer
from InputLayer import InputLayer
from Edges import Edges

np.random.seed(1)

class NeuralNetwork:
    def __init__(self, input_dimensions):
        self.n_layers = 1
        self.weights = []
        self.layers = np.array([InputLayer(input_dimensions)])
        
    def add_layer(self, dimensions):
        edges = Edges(dimensions, self.layers[self.n_layers - 1].dimensions)
        self.layers = np.append(self.layers, HiddenLayer(dimensions, edges, self.layers[self.n_layers - 1].a))
        self.weights = np.append(self.weights, edges)
        self.n_layers += 1
        
    def train(self, X, t, n_epoch, learning_rate = 1.0):
        for epoch in range(n_epoch):
            for x, _t in zip(X, t):
                # Forwardpropagation
                self._feedforward(x)
                
                # Backpropagation
                for l in reversed(range(1, self.n_layers)): # Runs through each layer (except input layer)
                    for j in range(self.layers[l].dimensions): # Runs through each node of layer l
                        if (l == self.n_layers - 1): # If layer l is the output layer
                            self.layers[l].neurons[j][0].grad = 2 * (self.layers[l].neurons[j][0].a - _t[j])
                            
                            for k in range(self.layers[l - 1].dimensions): 
                                self.weights[l - 1].weights[j, k] -= learning_rate * self.layers[l - 1].a[k][0] * self.layers[l].neurons[j][0].sigmoid_derived(self.layers[l].neurons[j][0].z) * self.layers[l].neurons[j][0].grad
                            
                            self.layers[l].neurons[j][0].b -= learning_rate * self.layers[l].neurons[j][0].sigmoid_derived(self.layers[l].neurons[j][0].z) * self.layers[l].neurons[j][0].grad
                        else:
                            for k in range(self.layers[l - 1].dimensions): 
                                self.layers[l].neurons[j][0].grad = 0
                                for _j in range(self.layers[l + 1].dimensions):
                                    self.layers[l].neurons[j][0].grad += self.weights[l].weights[_j, k] * self.layers[l + 1].neurons[_j][0].sigmoid_derived(self.layers[l + 1].neurons[_j][0].z) * self.layers[l + 1].neurons[_j][0].grad
                                self.weights[l - 1].weights[j, k] -= learning_rate * self.layers[l - 1].a[k][0] * self.layers[l].neurons[j][0].sigmoid_derived(self.layers[l].neurons[j][0].z) * self.layers[l].neurons[j][0].grad
                            
                            self.layers[l].neurons[j][0].b -= learning_rate * self.layers[l].neurons[j][0].sigmoid_derived(self.layers[l].neurons[j][0].z) * self.layers[l].neurons[j][0].grad
                            
    def _feedforward(self, x):
        self.layers[0].feedforward(x)
        for l in range(1, self.n_layers):
            self.layers[l].feedforward(self.weights[l - 1], self.layers[l - 1].a)
                            
    def predict(self, x):
        self.layers[0].predict(x)
        for l in range(1, self.n_layers):
            self.layers[l].predict(self.weights[l - 1], self.layers[l - 1].a)
            
        return self.layers[self.n_layers - 1].a