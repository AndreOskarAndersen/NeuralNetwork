import numpy as np
from NeuralNetwork import NeuralNetwork
import random

#random.seed(1) # Picking seed for debugging

# Creating model
model = NeuralNetwork(2)
model.add_layer(2)
model.add_layer(2)

# Generating data
X = np.array([[0, 0]]).reshape((1, -1))
y = np.array([[0, 0]]).reshape((1, -1))

for i in range(1000):
    rnd = random.randint(0, 3)
    if (rnd == 0):
        X = np.concatenate((X, [[0, 0]]), axis = 0)
        y = np.concatenate((y, [[0, 0]]), axis = 0)
    elif (rnd == 1):
        X = np.concatenate((X, [[1, 0]]), axis = 0)
        y = np.concatenate((y, [[1, 1]]), axis = 0)
    elif (rnd == 2):
        X = np.concatenate((X, [[0, 1]]), axis = 0)
        y = np.concatenate((y, [[1, 1]]), axis = 0)
    elif (rnd == 3):
        X = np.concatenate((X, [[1, 1]]), axis = 0)
        y = np.concatenate((y, [[0, 0]]), axis = 0)

# Training
model.train(X, y, 10, learning_rate = 0.1)

# Prediction 
print(model.predict([0, 0])) # Returns 0.02670295
print(model.predict([1, 1])) # Returns 0.02428404
print(model.predict([1, 0])) # Returns 0.96923397
print(model.predict([0, 1])) # Returns 0.97006255