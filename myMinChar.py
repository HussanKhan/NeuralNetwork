import numpy as np

############# Load in data #############
sampleData = open("input.txt", "r").read()

# Store characters
chars = set(sampleData) # Stores all non-dupcates from text, like letters and punc
chars = list(chars) # Makes set into list

# Data about data
dataSize = len(sampleData)
vocabSize = len(chars)

# Mappings for char and index for one-hot
charToIndex = {}
indexToChar = {}
# Filling Mapping
for index, char in enumerate(chars):
    charToIndex[char] = index
    indexToChar[index] = char

# Stats
print("Data has {} characters, {} unique.".format(dataSize, vocabSize))


############# Creating Network #############

# Hyperparamaters - network shape
hiddenSize = 100
seqLength = 25 # How many times steps for each input
learningRate = 0.01

# Model parameters - RNN shape
inputLayer = np.random.rand(hiddenSize, vocabSize)*0.01
hiddenHidden = np.random.rand(hiddenSize, hiddenSize)*0.01
hiddenOutput = np.random.rand(vocabSize, hiddenSize)*0.01

biasHidden = np.zeros_like((hiddenHidden, 1)) # hidden bias
biasOutput = np.zeros_like((vocabSize, 1)) # output bias

############## Training ##############

def train(inputs, targets, prevHiddenStates):

    # Stores each timestep for later backprop
    allInputs = {}
    allHiddenStates = {-1: prevHiddenStates}
    allOutputs = {}
    allProbabilties = {}

    # Stores total loss for backprop
    totalLoss = 0

    # Forward pass
    for x in range(len(inputs)):
       # One-hot input
       allInputs[x] = np.zeros((vocabSize, 1))
       allInputs[x][inputs[x]] = 1
       
       # Feed input to layer
       uX = np.dot(inputLayer, allInputs[x])

       # Prev Hidden(feeds last hidden) to Hidden
       hH = np.dot(hiddenHidden, allHiddenStates[x-1]) + biasHidden
        
       # Computes current hidden state, and adds it to dict
       allHiddenStates[x] = np.tanh(uX + hH)

       # Output from hidden to output
       allOutputs[x] = np.dot(hiddenOutput, allHiddenStates[x]) + biasOutput

       # Probabilties for next char
       # expo of all outputs / sum of all expos
       allProbabilties[x] = np.exp(allOutputs[x]) / np.sum(np.exp(allOutputs[x]))

       # Checks prob value for correct target
       # adds as loss usng negative loss
       # Softmax and calculate loss
       totalLoss += -np.log(allProbabilties[x][targets[x]][0])

    # Backward pass
    # go backwards through timestamps above
    deltaHiddenOutput = np.zeros_like(hiddenOutput)
    deltaInputLayer = np.zeros_like(inputLayer)
    deltaHiddenHidden = np.zeros_like(hiddenHidden)

    deltaBiasOutput = np.zeros_like(biasOutput)
    deltaBiasHidden = np.zeros_like(biasHidden)
    for b in range(len(inputs)-1, 0, -1):
        # Creates copy to do math on output
        deltaOutput = np.copy(allProbabilties[b])
        
        # Calculates error at that target position
        deltaOutput[targets[b]] -= 1

        # gradient for output layer
        deltaHiddenOutput += np.dot(deltaOutput, allHiddenStates[b].T)
        deltaBiasOutput += deltaHiddenOutput

        # Make gradient for out compatiable with hidden Layer
        # Carry gradient/loss through network 
        hiddenGrad = np.dot(hiddenOutput.T, deltaOutput)

        # error x tanh derivative
        deltaHiddenRaw = hiddenGrad * (1 - allHiddenStates[b]**2)
        deltaBiasHidden += deltaHiddenRaw

        # derivative for hidden layers
        deltaHiddenHidden += np.dot(deltaHiddenRaw, allHiddenStates[b-1].T)

        # derivative for input weights
        deltaInputLayer += np.dot(deltaHiddenRaw, allInputs[b].T)





        


    return 0



inputPosition = 0
currentIteration = 0

# Keep training until user stops
while True:

    # If end of input is reached, reset
    if inputPosition+seqLength >= dataSize or currentIteration == 0:
        # Resets RNN temp mem
        # Hidden states are only kept for one target at a time
        # as all those n inputs lead to that target
        tempMemory = np.zeros((hiddenSize, 1)) # Memory from all inputs for one target
        inputPosition = 0

    # Inputs and targets aligned by index
    # stores key for word and maintains order
    inputs = []
    targets = []

    for char in sampleData[inputPosition:inputPosition+seqLength]:
        newInput = charToIndex[char]
        inputs.append(newInput)

    for char in sampleData[inputPosition+1:inputPosition+seqLength+1]:
        newTarget = charToIndex[char]
        targets.append(newTarget)

    print(inputs)
    print(targets)

    print(tempMemory.shape)

    # feed data into model
    train(inputs, targets, tempMemory)

    # Move up data pointer and iteration
    inputPosition += seqLength
    currentIteration += 1

    break


