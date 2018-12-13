from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
from mypkg.nn.conv.mini_vggnet import MiniVGGNet
import matplotlib.pyplot as plt
import numpy as np


def step_decay(epoch):
    # learning rate scheduler
    # drop the learning rate every E epochs by factor F
    # using formula: lr = init_lr * F ^ ((1+epoch)/E)
    init_lr = 0.01
    F = 0.5
    E = 5
    # compute learning rate for current epoch
    alpha = init_lr * (F ** (np.floor((1+epoch) / E)))
    # return computed learning rate
    return float(alpha)


# load the training and testing data
print("[INFO] loading CIFAR-10 data")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# scale the data to range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# init the optimizer
print("[INFO] compiling model")
# define callbacks to be passed to model
callbacks = [LearningRateScheduler(step_decay)]
# SGD optimizer with Nesterov momentum
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# simple SGD optimizer with learning rate of 0.01
# sgd = SGD(lr=0.01)
# compile the network
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])

# train the network
print("[INFO] training network")
epochs = 200
batch_size = 64
H = model.fit(x=trainX, y=trainY,
              validation_data=(testX, testY), callbacks=callbacks,
              batch_size=batch_size, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network")
predY = model.predict(testX, batch_size=batch_size)

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
