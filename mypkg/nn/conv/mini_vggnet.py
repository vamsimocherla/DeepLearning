from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # init model along with input shape
        model = Sequential()
        input_shape = (height, width, depth)

        # batch normalization operates over channels
        # chan_dim indicates the index of channel axis
        # chan_dim = -1 indicates the last index
        chan_dim = -1

        if backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            # chan_dim = 1 indicates the first index
            chan_dim = 1

        # Mini VGGNet architecture => a shallower version of VGGNet model
        # ( CONV => RELU => BN => CONV => RELU => BN => POOL => DROPOUT ) * 2
        # ( FC => RELU => BN => DROPOUT ) * 1
        # FC => softmax

        # first set of CONV => RELU => BN => CONV => RELU => BN => POOL => DROPOUT
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # second set of CONV => RELU => BN => CONV => RELU => BN => POOL => DROPOUT
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # first set of FC => RELU => BN => DROPOUT
        # need to flatten input just before the first FC
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(0.50))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
