import os
from keras.preprocessing.image import img_to_array

# disable all TensorFlow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ImageProcessor:
    def __init__(self, data_format=None):
        # store the image data format
        self.data_format = data_format

    def process(self, image):
        # apply Keras utility function
        # to correctly arrange the dimensions of image
        return img_to_array(image, data_format=self.data_format)
