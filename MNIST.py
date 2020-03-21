import numpy as np
from keras.datasets import mnist
from NeuralNetwork import NeuralNetwork

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def normalize(x):
    return x / 255

def vectorize_y(y):
    z = np.zeros((10, 1))
    z[y] = 1
    return z

X_train = X_train[:1000]
X_train = X_train.reshape((X_train.shape[0], -1, 1))
X_train = np.apply_along_axis(normalize, 0, X_train)

y_train = y_train[:1000]
y_train = y_train.reshape(-1, 1)
y_train = np.apply_along_axis(vectorize_y, 1, y_train)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))

X_test = X_test.reshape((X_test.shape[0], -1, 1))
X_test = np.apply_along_axis(normalize, 0, X_test)

y_test = y_test.reshape(-1, 1)
y_test = np.apply_along_axis(vectorize_y, 1, y_test)
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))

model = NeuralNetwork(X_train.shape[1])
model.add_layer(64)
model.add_layer(10)

print("started training")
model.train(X_train, y_train, 40, 0.5)

for i in range(20):
    print("Prediction: {}, True: {}".format(np.argmax(model.predict(X_test[i])), np.argmax(y_test[i])))