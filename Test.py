import numpy as np
from NeuralNetwork import NeuralNetwork
import random

random.seed(1)

model = NeuralNetwork(2)
model.add_layer(2)
model.add_layer(1)

X = np.array([[0, 0]]).reshape((1, -1))
y = np.array([[0]]).reshape((1, -1))

print("Generating data...")

for i in range(1000):
    a = random.randint(0, 3)
    if (a == 0):
        X = np.concatenate((X, [[0, 0]]), axis = 0)
        y = np.concatenate((y, [[0]]), axis = 0)
    elif (a == 1):
        X = np.concatenate((X, [[1, 0]]), axis = 0)
        y = np.concatenate((y, [[1]]), axis = 0)
    elif (a == 2):
        X = np.concatenate((X, [[0, 1]]), axis = 0)
        y = np.concatenate((y, [[1]]), axis = 0)
    elif (a == 3):
        X = np.concatenate((X, [[1, 1]]), axis = 0)
        y = np.concatenate((y, [[0]]), axis = 0)

print("Fitting...")
model.train(X, y, 10, learning_rate = 1)


print("PREDICTIONS")
print(model.predict([0, 0]))
print(model.predict([1, 1]))
print(model.predict([1, 0]))
print(model.predict([0, 1]))