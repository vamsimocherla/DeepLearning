from mypkg.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend
import matplotlib.pyplot as plt
import numpy as np
import warnings

# get the full MNIST dataset
print("[INFO] loading MNIST dataset")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    dataset = datasets.fetch_mldata("MNIST Original", data_home="datasets/")

# reshape the input data base on channels ordering
if backend.image_data_format() == "channels_first":
    data = dataset.data.reshape(dataset.data.shape[0], 1, 28, 28)
else:
    data = dataset.data.reshape(dataset.data.shape[0], 28, 28, 1)

# scale the raw pixel intensities to range [0, 1]
data = data / 255.0
trainX, testX, trainY, testY = train_test_split(data, dataset.target.astype("int"),
                                                test_size=0.25, random_state=42)

# convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# init the optimizer
print("[INFO] compiling model")
# SGD optimizer with Nesterov momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# simple SGD optimizer with learning rate of 0.01
# sgd = SGD(lr=0.01)
# compile the network
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])

# train the network
print("[INFO] training network")
epochs = 20
H = model.fit(x=trainX, y=trainY,
              validation_data=(testX, testY),
              batch_size=128, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network")
predY = model.predict(testX, batch_size=128)

# convert label probabilities to corresponding labels
# with highest prediction probability
predY = predY.argmax(axis=1)
testY = testY.argmax(axis=1)
# print classification report
print(classification_report(y_true=testY,
                            y_pred=predY,
                            target_names=[str(x) for x in lb.classes_]))

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
plt.show()
