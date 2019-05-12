import numpy
# used to import sigmoid function
import scipy.special 

class NeuralNet():
    
    def __init__(self, learningRate, hiddenSize):
        self.learningRate = learningRate # Learning Rate
        self.currentLayers = 1
        self.hiddenSize = hiddenSize # Number of nodes in hidden layer
        self.netMap = {} # Holds weight mappings
        # activation function
        self.activation = lambda x: scipy.special.expit(x)

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
        inputData = inputData
        
        for n in range(1, self.currentLayers):
            layerOutput = numpy.dot(self.netMap[n]["weights"], inputData)
            inputData = self.activation(layerOutput)
            self.netMap[n]["output"] = inputData

        finalOutput = inputData

        return finalOutput

    def backPropagation(self, targets):
        
        # changeWeight = -lr ( [Ek * Ok (1 - Ok) * Oj] )

        finalOutput = self.netMap[self.currentLayers - 1]["output"]
        outputError = targets - finalOutput

        for l in range(self.currentLayers - 1, 1, 1):
            if (l - 1) != 0:
                currentError = outputError
                currentOutput = self.netMap[l]["output"]

                prevOutput = numpy.transpose(self.netMap[l-1]["output"])

                self.netMap[l]["weights"] += -(self.learningRate) * numpy.dot((currentError * currentOutput * (1 - currentError)) ,prevOutput)

                prevError = numpy.dot(numpy.transpose(self.netMap[l]["weights"]), currentError)
                outputError = prevError 


    def trainNetwork(self):
        for n in range(1, self.currentLayers):
            print(n)
            print(self.netMap[n])
            print("weights ", self.netMap[n]["weights"].shape)
            print("output ", self.netMap[n]["output"].shape)
            print("\n")

# print((numpy.random.rand(3, 2) - 0.5))

net = NeuralNet(learningRate=0.5, hiddenSize=4)

net.addLayer(inputLayer=True, inputSize=5)
net.addLayer()
net.addLayer()
net.addLayer(outputLayer=True, outputSize=2)
inputData = (numpy.random.rand(5, 1) - 0.5)
# print(net.feedForward(inputData))
net.feedForward(inputData)
net.trainNetwork()

