import numpy as np
from NeuralNetwork import NeuralNetwork
import random

model = NeuralNetwork(2)
model.add_layer(2)
model.add_layer(1)

data_1 = np.array([0, 1, 1]).reshape((1, -1))

data_2 = np.array([1, 0, 1]).reshape((1, -1))

data_3 = np.array([0, 0, 0]).reshape((1, -1))

data_4 = np.array([1, 1, 0]).reshape((1, -1))

data = np.array(data_1).reshape((1, -1))

for i in range(100):
    a = random.randint(0, 3)
    if (a == 0):
        data = np.concatenate((data, data_1), axis = 0)
    elif (a == 1):
        data = np.concatenate((data, data_2), axis = 0)
    elif (a == 2):
        data = np.concatenate((data, data_3), axis = 0)
    elif (a == 3):
        data = np.concatenate((data, data_4), axis = 0)

X = data[:, :2]
y = data[:, 2].reshape((-1, 1))

model.train(X, y, 100, learning_rate = 1.0)


print(model.predict(data_1[0][:2]))
print(model.predict(data_2[0][:2]))
print(model.predict(data_3[0][:2]))
print(model.predict(data_4[0][:2]))