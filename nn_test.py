from mypkg.nn import NeuralNetwork as nn
import numpy as np


def test_preceptron(X, y, name=None):
    # 1 input layer with 2 input nodes
    # 1 hidden layer with 2 nodes
    # 1 output layer with 1 node --> prediction
    layers = [2, 2, 1]
    # init new NeuralNetwork instance
    network = nn.NeuralNetwork(layers, alpha=0.5)
    # fit the data to a model
    network.fit(X, y, epochs=1000, display=500)

    # make predictions
    print("-" * 20)
    print("Dataset: {}".format(name))
    for x, target in zip(X, y):
        # get the prediction
        pred = network.predict(x, add_bias=True)
        step = 1 if pred[0][0] > 0.5 else 0
        print("input: {} actual: {} predicted: {} step: {}".format(x, target[0], pred, step))
    print("-" * 20)


def dataset_or():
    # input for OR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    # test our perceptron
    test_preceptron(X, y, name="OR")


def dataset_and():
    # input for AND dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    # test our perceptron
    test_preceptron(X, y, name="AND")


def dataset_xor():
    # input for XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    # test our perceptron
    test_preceptron(X, y, name="XOR")


# test our datasets
dataset_or()
dataset_and()
dataset_xor()
