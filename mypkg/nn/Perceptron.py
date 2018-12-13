import numpy as np


class Perceptron:
    def __init__(self, N=0, alpha=0.1):
        # init the number of input features
        self.N = N
        # init the weight vector
        # zero mean and unit variance
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        # init the learning rate
        self.alpha = alpha

    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # add bias parameter to the input data
        # this is known as 'bias trick'
        # allows us to treat bias as a trainable parameter
        # do not need to explicitly track the bias parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop for number of epochs
        for epoch in np.arange(0, epochs):
            # for each epoch compute the loss over X
            # loop over each data point
            for (x, target) in zip(X, y):
                # compute the dot product between input x and weight vector
                # get prediction by passing through the step function
                pred = self.step(np.dot(x, self.W))

                # if prediction is wrong
                if pred != target:
                    # compute the error
                    error = pred - target
                    # update the weight vector
                    self.W += -self.alpha * error * x

    def predict(self, X, add_bias=True):
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        # check if bias column needs to be added
        # just like we did in the training phase
        if add_bias:
            # augment a column of 1's to the feature matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # compute the dot product between input x and weight vector
        # get prediction by passing through the step function
        pred = self.step(np.dot(X, self.W))
        return pred
