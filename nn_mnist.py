from mypkg.nn import NeuralNetwork as nn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print("loading MNIST dataset")
# load MNIST dataset
digits = datasets.load_digits()
# convert the data type to float
data = digits.data.astype("float")
# apply min/max scaling to scale the
# pixel intensity values to range [0, 1]
data = (data - data.min()) / (data.max() - data.min())
# each image is 8 x 8 = 64 dimension feature vector
print("num_samples: {}, dimension: {}".format(data.shape[0], data.shape[1]))

# construct training and testing split
trainX, testX, trainY, testY = train_test_split(data, digits.target, test_size=0.25)

# convert the labels from integers to vectors:
# 5 -> [0 0 0 0 0 1 0 0 0 0]
# 9 -> [0 0 0 0 0 0 0 0 0 1]
# this is called one-hot encoding
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train our network
print("training network")
# num input nodes = num input features
input_nodes = trainX.shape[1]
# num output nodes = num prediction classes
output_nodes = trainY.shape[1]
network = nn.NeuralNetwork(layers=[input_nodes, 32, 16, output_nodes])
print("network: {}".format(network))
network.fit(trainX, trainY, epochs=1000, display=100)

# evaluate the network performance
print("evaluating network")
# compute the predictions of test data
# this returns a vector of probabilities for each class
# pred = [ list of prob that x belongs to class c ]
predY = network.predict(testX)
# need to get the class label with highest probability
predY = predY.argmax(axis=1)
testY = testY.argmax(axis=1)
# print detailed performance report
print(classification_report(testY, predY))
