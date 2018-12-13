from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # init the model
        model = Sequential()
        input_shape = (height, width, depth)

        # if using 'channels first', update input shape
        if backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # original LeNet architecture consists of:
        # INPUT => (CONV => TANH => POOL) => (CONV => TANH => POOL) => (FC => TANH) => (FC => softmax)
        # replace activation function TanH with ReLU
        # INPUT => (CONV => RELU => POOL) => (CONV => RELU => POOL) => (FC => RELU) => (FC => softmax)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first set of FC => RELU
        # input needs to be flattened before passing to FC layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
