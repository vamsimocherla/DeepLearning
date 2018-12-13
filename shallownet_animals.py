import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from mypkg.nn.conv.shallownet import ShallowNet
# from nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# disable all TensorFlow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
                default="/home/vamsimocherla/Research/DeepLearning/DrAdrian/0-StarterBundle/SB_Code/datasets/animals",
                help="path to input dataset")
args = vars(ap.parse_args())
# grab the list of images that we’ll be describing
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

# initialize the image processors
sp = SimplePreprocessor(32, 32)
ip = ImageToArrayPreprocessor()

# load dataset
sdl = SimpleDatasetLoader(preprocessors=[sp, ip])
# scale raw pixel intensities to range [0, 1]
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

# split data into training and testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
# one-hot encoding - convert labels to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model")
opt = SGD(lr=0.05)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
# train the network
print("[INFO] training network")
epochs = 100
H = model.fit(x=trainX, y=trainY,
              validation_data=(testX, testY),
              batch_size=32, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network")
predY = model.predict(testX, batch_size=32)
# get the labels corresponding to the highest probability
testY = testY.argmax(axis=1)
predY = predY.argmax(axis=1)
print(classification_report(y_true=testY,
                            y_pred=predY,
                            target_names=["cat", "dog", "panda"]))

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
# plt.savefig(args["output"])
plt.show()
