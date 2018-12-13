import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # init an empty weight matrix
        # these weights are randomly sampled and then normalized
        self.W = []
        # init the network layers
        self.layers = layers
        # init the learning rate
        self.alpha = alpha
        # init the weight matrices
        self.init_weights()

    def init_weights(self):
        # loop over all the layers except last 2
        # this is due to the output layer not needing a bias term
        for i in np.arange(0, len(self.layers) - 2):
            # init a weight matrix with size M x N
            # this weight matrix corresponds to all the weights between layer[i] and layer[i+1]
            # since this is a fully connected network, we need M x N weights
            # +1 term is for the bias trick
            w = np.random.randn(self.layers[i] + 1, self.layers[i+1] + 1)
            # weights are randomly chosen from a normal distribution
            # divide the weights by sqrt(number of nodes)
            # to normalize the variance of each neuron's output
            self.W.append(w / np.sqrt(self.layers[i]))
        # last 2 layers is a special case
        # output layer does not need a bias term
        # the layer before the output layer needs a bias term
        # layers[-2] gives the last but one layer
        # layers[-1] gives the last layer - which is the output layer
        w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        self.W.append(w / np.sqrt(self.layers[-2]))

    def fit(self, X, y, epochs=1000, display=100):
        # augment the bias term - the bias trick
        # allows us to treat the bias term as a trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over epochs
        for epoch in np.arange(0, epochs):
            # loop over each data input
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # display loss for every 'display' epochs
            if (epoch + 1) % display == 0:
                loss = self.compute_loss(X, y)
                print("epochs: {} loss: {:.7f}".format(epoch+1, loss))

    def fit_partial(self, x, y):
        # construct a list of output activations at each layer
        # the first activation is the input vector itself
        A = [np.atleast_2d(x)]

        ######################
        # FEED FORWARD PHASE #
        ######################
        # compute output activations for each layer
        for layer in np.arange(0, len(self.layers)-1):
            # feed forward the activation at the current layer
            # by computing the dot product of activation and weight matrix
            # this is called as the net input to the current layer
            net = np.dot(A[layer], self.W[layer])

            # pass the net input through the activation function
            # this is called the net output
            out = self.sigmoid(net)

            # once we have the net output, append to the list of activations
            # this net output will be the activation for the next layer
            A.append(out)

        ##########################
        # BACK PROPAGATION PHASE #
        ##########################
        # error is computed as E = (1/2) * (target - out)^2
        # the first phase of backpropagation is to compute the diff between
        # out prediction (final output activation) and the true target value
        # 1 => derivative( error w.r.t out ) = ( out - target )
        # 2 => derivative( out w.r.t net ) = sigmoid(net) * (1 - sigmoid(net))
        # 3 => derivative( error w.r.t net ) = (1) * (2)
        error = A[-1] - y
        # first entry of deltas = the error of output layer * derivative of activation
        D = [error * self.sigmoid_deriv(A[-1])]

        # from here we need to apply the chain rule
        for layer in np.arange(len(A) - 2, 0, -1):
            # delta_partial = delta of previous layer (DOT) weight matrix of current layer
            # delta for the current layer = delta_partial * derivative of activation of current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            # append to the list of derivatives
            D.append(delta)

        # reverse the deltas since we looped in the reverse order
        D = D[::-1]

        #######################
        # WEIGHT UPDATE PHASE #
        #######################
        for layer in np.arange(0, len(self.W)):
            # update the weights by computing the DOT between activation
            # with the corresponding delta, then scaling by a 'learning rate'
            # in the negative gradient direction
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        # initialize the output prediction as input features
        # this will be forward propagated through the network
        X = np.atleast_2d(X)

        if add_bias:
            # insert a columns of 1's as the bias term - the bias trick
            X = np.c_[X, np.ones((X.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            # computing the output prediction will be taking the dot product between
            # the current activation value and weight matrix of current layer
            # then passing it through the non-linear activation function

            # feed forward the activation at the current layer
            # by computing the dot product of activation and weight matrix
            # this is called as the net input to the current layer
            net = np.dot(X, self.W[layer])
            # pass the net input through the activation function
            # this is called the net output
            out = self.sigmoid(net)
            # the net output of the current layer will be the net input for the next
            X = out

        return X

    def compute_loss(self, X, y):
        # make predictions for input data
        y = np.atleast_2d(y)
        predictions = self.predict(X, add_bias=False)
        # compute loss = (1/2) * (target - predictions) ^ 2
        loss = (1/2) * np.sum((y - predictions) ** 2)
        return loss

    def __repr__(self):
        # return a string representation of the class instance
        return "NeuralNetwork: {}".format("-".join(str(layer) for layer in self.layers))

    def sigmoid(self, x):
        # compute the sigmoid activation value of the input x
        # sig(x) = 1 / (1 + exp^-x)
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the differential of sigmoid input x
        # diff ( sig(x) ) = x * (1 - x)
        return x * (1 - x)
