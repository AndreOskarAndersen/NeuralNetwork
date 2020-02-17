import numpy as np
from NeuralNetwork import NeuralNetwork
import random

random.seed(1) # Picking seed for debugging

# Creating model
model = NeuralNetwork(2)
model.add_layer(2)
model.add_layer(1)

# Generating data
X = np.array([[0, 0]]).reshape((1, -1))
y = np.array([[0]]).reshape((1, -1))

for i in range(10000):
    rnd = random.randint(0, 3)
    if (rnd == 0):
        X = np.concatenate((X, [[0, 0]]), axis = 0)
        y = np.concatenate((y, [[0]]), axis = 0)
    elif (rnd == 1):
        X = np.concatenate((X, [[1, 0]]), axis = 0)
        y = np.concatenate((y, [[1]]), axis = 0)
    elif (rnd == 2):
        X = np.concatenate((X, [[0, 1]]), axis = 0)
        y = np.concatenate((y, [[1]]), axis = 0)
    elif (rnd == 3):
        X = np.concatenate((X, [[1, 1]]), axis = 0)
        y = np.concatenate((y, [[0]]), axis = 0)

# Training
model.train(X, y, 100, learning_rate = 1.0)

# Prediction 
print(model.predict([0, 0])) # Returns 0.03126419
print(model.predict([1, 1])) # Returns 0.794184
print(model.predict([1, 0])) # Returns 0.79322682
print(model.predict([0, 1])) # Returns 0.79121434