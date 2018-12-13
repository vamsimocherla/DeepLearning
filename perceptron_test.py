from mypkg.nn import Perceptron as p
import numpy as np


def test_preceptron(X, y, name=None):
    # init new Perceptron instance
    perceptron = p.Perceptron(X.shape[1], alpha=0.1)
    # fit the data to a model
    perceptron.fit(X, y, epochs=20)

    # make predictions
    print("-" * 20)
    print("Dataset: {}".format(name))
    for x, target in zip(X, y):
        # get the prediction
        pred = perceptron.predict(x, add_bias=True)
        print("input: {} actual: {} predicted: {}".format(x, target[0], pred))
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
