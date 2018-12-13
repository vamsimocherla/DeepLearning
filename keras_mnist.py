import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

# disable all TensorFlow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output",
                required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# get the full MNIST dataset
print("[INFO] loading MNIST dataset")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    dataset = datasets.fetch_mldata("MNIST Original", data_home="datasets/")

# scale the raw pixel intensities to range [0, 1]
data = dataset.data.astype("float") / 255.0
trainX, testX, trainY, testY = train_test_split(data, dataset.target, test_size=0.25)

# convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# define our network using Keras implementation
# sequential indicates that our network will be feed forward type
# and layers will be added sequentially one on top of the other
network = Sequential()
# Dense class indicates that the network is fully connected
# add the network layers --> 784-256-128-10 architecture using Keras
input_nodes = (784, )
output_nodes = 10
network.add(Dense(265, input_shape=input_nodes, activation="sigmoid"))
network.add(Dense(128, activation="sigmoid"))
network.add(Dense(output_nodes, activation="softmax"))

# train the network model using SGD - Stochastic Gradient Descent
# SGD class is the optimization method
print("[INFO] training network")
# SGD optimizer with Nesterov momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# simple SGD optimizer with learning rate of 0.01
# sgd = SGD(lr=0.01)
# use cross entropy loss function for evaluation
network.compile(loss="categorical_crossentropy",
                optimizer=sgd,
                metrics=["accuracy"])
# epochs for training the network
epochs = 200
# returns a dictionary with loss/accuracy of network
H = network.fit(x=trainX,
                y=trainY,
                validation_data=(testX, testY),
                epochs=epochs,
                verbose=1,
                batch_size=128)

# evaluate the network
print("[INFO] evaluating network")
predY = network.predict(testX, batch_size=128)
# convert the returned probability vectors to
# class labels with highest probability
predY = predY.argmax(axis=1)
testY = testY.argmax(axis=1)
# print the classification report
print(classification_report(testY,
                            predY,
                            target_names=[str(l) for l in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig(args["output"])
