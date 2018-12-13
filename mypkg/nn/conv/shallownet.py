import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend

# disable all TensorFlow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        # input shape to be 'channels last' as in keras.json config
        input_shape = (height, width, depth)

        # if config is 'channels first'
        if backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # add layers to the network
        # CONV => RELU => FC => softmax
        # num filters for this level
        K = 32
        # receptive field of the filter - size of the filter
        F = (3, 3)
        model.add(Conv2D(K, F,
                         # indicates with zero padding
                         padding="same",
                         # shape of input vector
                         input_shape=input_shape))
        # add an ReLU layer to the CONV output
        model.add(Activation("relu"))

        # Flatten classes converts the multi-dimensional input volume
        # and flattens it into a 1D vector prior to feeding to FC layers
        model.add(Flatten())
        # add a fully connected ( FC ) layer before the softmax activation
        # number of nodes in FC layer = number of classes of dataset
        model.add(Dense(classes))
        # softmax classifier - gives label probabilities for each class
        model.add(Activation("softmax"))

        return model
