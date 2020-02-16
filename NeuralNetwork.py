import numpy as np
from HiddenLayer import HiddenLayer
from InputLayer import InputLayer
from Edges import Edges

np.random.seed(1) # Picking seed for debugging

class NeuralNetwork:
    def __init__(self, input_dimensions):
        self.n_layers = 1
        self.weights = []
        self.layers = np.array([InputLayer(input_dimensions)])
        
    def add_layer(self, dimensions):
        self.weights = np.append(self.weights, Edges(dimensions, self.layers[self.n_layers - 1].dimensions))
        self.layers = np.append(self.layers, HiddenLayer(dimensions))
        self.n_layers += 1
        
    @staticmethod
    def sigmoid_derived(x):
        return np.multiply(x, 1.0 - x)
        
    def train(self, X, t, n_epoch, learning_rate = 1.0):
        def update_weights(l, j, k):
            self.weights[l - 1].weights[j, k] -= learning_rate * self.layers[l - 1].a[k][0] * NeuralNetwork.sigmoid_derived(self.layers[l].neurons[j][0].a) * self.layers[l].neurons[j][0].grad
            
        def update_bias(l, j):
            self.layers[l].neurons[j][0].b -= learning_rate * NeuralNetwork.sigmoid_derived(self.layers[l].neurons[j][0].a) * self.layers[l].neurons[j][0].grad
        
        for _ in range(n_epoch):
            for x, _t in zip(X, t):
                
                # Forwardpropagation
                self._forwardprop(x)
                
                # Backpropagation
                for l in reversed(range(1, self.n_layers)): # Iterates through each layer (except input layer)
                    for j in range(self.layers[l].dimensions): # Iterates through each node of layer l
                        self.layers[l].neurons[j][0].grad = 0
                        
                        for k in range(self.layers[l - 1].dimensions):
                            if (l == self.n_layers - 1): # If layer l is the output layer
                                self.layers[l].neurons[j][0].grad = 1/len(t[0]) * 2 * (self.layers[l].neurons[j][0].a - _t[j])
                            else:
                                for _j in range(self.layers[l + 1].dimensions):
                                    self.layers[l].neurons[j][0].grad += self.weights[l].weights[_j, k] * NeuralNetwork.sigmoid_derived(self.layers[l + 1].neurons[_j][0].a) * self.layers[l + 1].neurons[_j][0].grad
                                    
                            update_weights(l, j, k)
                                
                        update_bias(l, j)
                            
    def _forwardprop(self, x):
        self.layers[0].feedforward(x)
        for l in range(1, self.n_layers):
            self.layers[l].feedforward(self.weights[l - 1], self.layers[l - 1].a)
                            
    def predict(self, x):
        self._forwardprop(x)
            
        return self.layers[self.n_layers - 1].a