import numpy
# used to import sigmoid function
import scipy.special 
# used for plotting
import matplotlib.pyplot as plt

from NeuralOne import NeuralOne

# Loading training data
data = open("mnist_train.csv", "r")
trainInfo = data.readlines() # all data in file, first index is target value
data.close()

# View Data
# pictureData = trainInfo[0].split(",")[1:] # First index is target value
# reShaped = numpy.asfarray(pictureData).reshape((28, 28)) # 782 = 28 * 28 - total entries
# scaled = (((reShaped) / 255.0) * 0.99) + 0.01 # Scaled from 0-255, to 0.01-1.00
# plt.imshow(scaled)
# plt.show()

# Format Train Data Input
trainData = []
targetData = []

for d in trainInfo:
    # Extracting array data
    arrayData = d.split(",")

    # inputData = numpy.array(arrayData[1:], ndmin=2).T
    scaled = (((numpy.asfarray(arrayData[1:])) / 255.0) * 0.99) + 0.01 # Scaled from 0-255, to 0.01-1.00
    
    # Creating target data
    targets = numpy.zeros(10) + 0.01
    label = int(arrayData[0])
    targets[label] = 0.99

    trainData.append(scaled)
    targetData.append(targets)

# Loading training data
data = open("mnist_test.csv", "r")
testInfo = data.readlines() # all data in file, first index is target value
data.close()

# Format Test Data Input
testData = []
testTargetData = []

for d in testInfo:
    # Extracting array data
    arrayData = d.split(",")
    scaled = (((numpy.asfarray(arrayData[1:])) / 255.0) * 0.99) + 0.01 # Scaled from 0-255, to 0.01-1.00

    # Creating target data
    targets = numpy.zeros(10) + 0.01
    label = int(arrayData[0])
    targets[label] = 0.99

    testData.append(scaled)
    testTargetData.append(targets)

net = NeuralOne(learningRate=0.1, hiddenSize=300)
net.addLayer(inputLayer=True, inputSize=784)
net.addLayer(outputLayer=True, outputSize=10)
net.trainNetwork(trainData, targetData, epochs=6)
net.testNetwork(trainData, targetData)

# View Data
print(testTargetData[1])
pictureData = net.feedBackward(testTargetData[1])
reShaped = numpy.asfarray(pictureData).reshape((28, 28)) # 782 = 28 * 28 - total entries
plt.imshow(reShaped)
plt.show()

# testArr = numpy.random.rand(10, 1)

# print(testArr.shape)
# print(testArr)
# print(testArr.transpose())