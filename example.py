import numpy
import matplotlib.pyplot as plt
# from NeuralOne import NeuralOne
from NTEST import NeuralOne

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

net = NeuralOne(learningRate=0.001, hiddenSize=100)
# net.loadNeuralNet("mnist_net")
net.addLayer(inputLayer=True, inputSize=784)
net.addLayer()
net.addLayer()
net.addLayer(outputLayer=True, outputSize=10)
net.trainNetwork(trainData, targetData, epochs=6)
net.testNetwork(testData, testTargetData)
net.saveNeuralNet("mnist_net")

# Backwards feedback to see what network thinks
while True:
    testTar = input("BackTest Target: ")
    if testTar != "q":
        
        testTar = int(testTar)
        
        targets = numpy.zeros(10) + 0.01
        targets[testTar] = 0.99

        pictureData = net.feedBackward(targets)
        reShaped = numpy.asfarray(pictureData).reshape((28, 28)) # 782 = 28 * 28 - total entries
        plt.imshow(reShaped, cmap='Greys', interpolation='None')
        # View Data
        plt.show()
    else:
        break

# Basic prediction
while True:
    testTar = input("Predict: ")
    if testTar != "q":
        testTar = int(testTar)
        print(numpy.argmax(targetData[testTar]))
        pictureData = net.predict(trainData[testTar])
        print(pictureData)
    else:
        break



# testArr = numpy.random.rand(10, 1)

# print(testArr.shape)
# print(testArr)
# print(testArr.transpose())