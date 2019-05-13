import numpy
# used to import sigmoid function
import scipy.special 
from tqdm import tqdm

class NeuralOne():
    
    def __init__(self, learningRate, hiddenSize):
        self.learningRate = learningRate # Learning Rate
        self.currentLayers = 1
        self.hiddenSize = hiddenSize # Number of nodes in hidden layer
        self.netMap = {} # Holds weight mappings
        # activation function
        self.activation = lambda x: scipy.special.expit(x)
        self.reverseActivation = lambda x: scipy.special.logit(x)
        pass

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

    def feedForward(self, inputData):
        # W * I - Inputs feed into Weights
        inputData = numpy.array(inputData, ndmin=2).T
        
        for n in range(1, self.currentLayers):
            layerOutput = numpy.dot(self.netMap[n]["weights"], inputData)
            inputData = self.activation(layerOutput)
            self.netMap[n]["output"] = inputData

        finalOutput = inputData
        return finalOutput

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

            # Update weights based on graient descent, chain rule of sigmoid
            # Adding because error gives us sign, (See Perceptron Training by Udacity on Youtube)
            self.netMap[l]["weights"] += ((self.learningRate) * numpy.dot((newError * (currentOutput * (1 - currentOutput))) , prevOutput)) 
            
            # Spread error to weigths
            newError = numpy.dot(self.netMap[l]["weights"].T, newError)

    def feedBackward(self, target):
        # W.T * T - Feeds Target to weights in reverse order
        target = self.reverseActivation(numpy.array(target, ndmin=2).T)
        
        for i in range(self.currentLayers - 1, 0, -1):
            layerOutput = numpy.dot(self.netMap[i]["weights"].T, target)
            print(layerOutput.shape)
            target = layerOutput
        return target

    def trainNetwork(self, inputData, targetData, epochs):
        for i in range(1, epochs):
            print("EPOCH ", i)
            for index, value in tqdm(enumerate(inputData)):
                self.feedForward(value)
                self.backPropagation(targetData[index], value)

    def testNetwork(self, inputData, targetData):
        total_test = 0
        correct = []
        for index, value in enumerate(inputData):

            output = self.feedForward(value)
            
            # correct index, argmax returns index of highest value
            correctLabel = int(numpy.argmax(targetData[index]))

            # Prediction from network, index of highest prob
            prediction = numpy.argmax(output)

            # if it matches, append correct list
            if prediction == correctLabel:
                correct.append(1)

            total_test = total_test + 1
        
        print("Performance: {}".format(len(correct)/total_test))
        print("Out of {} tests, {} predictions were correct".format(total_test, len(correct)))
