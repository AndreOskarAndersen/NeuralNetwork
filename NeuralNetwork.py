import numpy as np
from HiddenLayer import HiddenLayer
from InputLayer import InputLayer

#np.random.seed(1) # Picking seed for debugging

class NeuralNetwork:
    def __init__(self, input_dimensions):
        self.n_layers = 1
        self.layers = np.array([InputLayer(input_dimensions)])
        
    def add_layer(self, dimensions):
        self.layers = np.append(self.layers, HiddenLayer(dimensions, self.layers[self.n_layers - 1].dimensions))
        self.n_layers += 1
        
    @staticmethod
    def sigmoid_derived(x):
        return np.multiply(x, 1.0 - x)
    
    def find_error(self, prediction, true_value):
        return np.mean(np.power(np.subtract(prediction, true_value), 2))
        
    def train(self, X, t, n_epoch, learning_rate = 1.0):
        def update_weights(l, j, k):
            self.layers[l].weights[j, k] -= learning_rate * self.layers[l - 1].a[k] * NeuralNetwork.sigmoid_derived(self.layers[l].a[j]) * self.layers[l].grad[j]
            
        def update_bias(l, j):
            self.layers[l].b[j] -= learning_rate * NeuralNetwork.sigmoid_derived(self.layers[l].a[j]) * self.layers[l].grad[j]
        
        for epoch in range(n_epoch):
            for x, _t in zip(X, t):
                
                # Forwardpropagation
                self._forwardprop(x)
                
                # Backpropagation
                for l in reversed(range(1, self.n_layers)): # Iterates through each layer (except input layer)
                    for j in range(self.layers[l].dimensions): # Iterates through each node of layer l
                        if (l == self.n_layers - 1): # If layer l is the output layer
                            self.layers[l].grad[j] = -2 * (_t[j] - self.layers[l].a[j])
                        else:
                            self.layers[l].grad[j] = 0
                            for _j in range(self.layers[l + 1].dimensions):
                                self.layers[l].grad[j] += 1/2 * self.layers[l + 1].weights[_j, j] * NeuralNetwork.sigmoid_derived(self.layers[l + 1].a[_j]) * self.layers[l + 1].grad[_j]
                                    
                                    
                for l in reversed(range(1, self.n_layers)):
                    for j in range(self.layers[l].dimensions):
                        for k in range(self.layers[l - 1].dimensions):
                            update_weights(l, j, k)
                        update_bias(l, j)
           
            if (epoch % 10 == 0):
                print("Epoch {} done. Error: {}".format(epoch, self.find_error(self.predict(x), _t)))          
                            
    def _forwardprop(self, x):
        self.layers[0].feedforward(x)
        for l in range(1, self.n_layers):
            self.layers[l].feedforward(self.layers[l].weights, self.layers[l - 1].a)
        
    def predict(self, x):
        self._forwardprop(x)
            
        return self.layers[self.n_layers - 1].a