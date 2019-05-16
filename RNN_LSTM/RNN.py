import numpy
# used to import sigmoid function
import scipy.special 
from tqdm import tqdm
import json

class RNN():
    
    def __init__(self, learningRate=0.001, hiddenSize=1):
        self.learningRate = learningRate # Learning Rate
        self.currentLayers = 1
        self.hiddenSize = hiddenSize # Number of nodes in hidden layer
        self.netMap = {} # Holds weight mappings
        # activation function
        self.activation = lambda x: numpy.tanh(x)
        self.reverseActivation = lambda x: scipy.special.logit(x)
        pass

    # Adds layers to netMap (Network Map)
    def addLayer(self, inputSize=0, outputSize=0, inputLayer=False, outputLayer=False):
        
        if inputLayer:
            # from input to hidden R x C 300 x 782, input = 782 x 1 
            # Output to next layer = 300 x 1
            inputHidden = (numpy.random.rand(self.hiddenSize, inputSize) - 0.5)
            self.netMap[self.currentLayers] = {"weights": inputHidden}
        elif outputLayer:
            # from hidden to output R x C 10 x 300, input = 300 x 1
            # Final out 10 x 1 
            hiddenOutput = (numpy.random.rand(outputSize, self.hiddenSize) - 0.5)
            self.netMap[self.currentLayers] = {"weights": hiddenOutput}
        else:
            # hidden sandwich layer R x C 300 x 300, input = 300 x 1,
            # W * I - 300 x 300 * 300 x 1
            # Output is 300 x 1
            hiddenLayer = (numpy.random.rand(self.hiddenSize, self.hiddenSize) - 0.5)
            self.netMap[self.currentLayers] = {"weights": hiddenLayer}

        self.currentLayers = self.currentLayers + 1 # Increments layer

    # Feeds forward through network, returns output
    def feedForward(self, inputData, prevOutput=False):
        
        # W * I - Inputs feed into Weights
        inputData = numpy.array(inputData, ndmin=2).T
        
        if prevOutput:
            inputData = inputData + prevOutput
        
        for n in range(1, self.currentLayers):
            layerOutput = numpy.dot(self.netMap[n]["weights"], inputData)
            inputData = self.activation(layerOutput)
            self.netMap[n]["output"] = inputData

        finalOutput = inputData
        
        return finalOutput

    # Returns output matrix from neural net
    def predict(self, inputData):
        result = self.feedForward(inputData)
        return result

    # BackProp through network, updates weights
    def backPropagation(self, targets, inputs):
        
        # changeWeight = -lr ( [Ek * Ok (1 - Ok) * Oj] )

        finalOutput = self.netMap[self.currentLayers - 1]["output"]
        targets = numpy.array(targets, ndmin=2).T
        
        # Gives us sign we need to update weights
        newError = targets - finalOutput

        for l in range(self.currentLayers - 1, 0, -1):

            currentOutput = self.netMap[l]["output"]

            if (l - 1) > 0:
                prevOutput = self.netMap[l-1]["output"].T
            else:
                prevOutput = numpy.array(inputs, ndmin=2)

            # Update weights based on graient descent, chain rule of tanh
            # Adding because error gives us sign, (See Perceptron Training by Udacity on Youtube)
            self.netMap[l]["weights"] += ((self.learningRate) * numpy.dot((newError * ( 1 - currentOutput**2 )) , prevOutput)) 

            # Spread error to weigths
            newError = numpy.dot(self.netMap[l]["weights"].T, newError)

        return finalOutput

    # Trains network using input and targets
    def trainNetwork(self, inputData, targetData, epochs):

        for i in range(1, epochs):

            print("EPOCH ", i)
            lastOutput = False
            
            for index, value in tqdm(enumerate(inputData)):
                self.feedForward(value, lastOutput)
                self.backPropagation(targetData[index], value)

    # Saves network as json
    def saveNeuralNet(self, name):

        formatNet = dict(self.netMap)

        for i, w in formatNet.items():

            storeW = formatNet[i]["weights"].tolist()
            storeO = formatNet[i]["output"].tolist()
            
            formatNet[i]["weights"] =  storeW
            formatNet[i]["output"] =  storeO

        with open("{}.json".format(name), "w") as file:
            json.dump({"neuralNet": formatNet}, file)

        self.loadNeuralNet(name)
    
    # Loads saved network
    def loadNeuralNet(self, name):
        
        with open("{}.json".format(name), "r") as file:
            data = json.load(file)
            file.close()

        jsonNet = data["neuralNet"]
        savedNet = {}
        
        for i, w in jsonNet.items():

            newWeight = jsonNet[i]["weights"]
            newOutput = jsonNet[i]["output"]

            i = int(i)

            savedNet[i] = {}
            savedNet[i]["weights"] =  numpy.array(newWeight)
            savedNet[i]["output"] =  numpy.array(newOutput)
            
            self.currentLayers = i + 1

        self.netMap = savedNet
        