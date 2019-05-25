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

biasHidden = np.zeros((hiddenSize, 1)) # hidden bias
biasOutput = np.zeros((vocabSize, 1)) # output bias

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
       # 100 x 65 * 65 x 1 = 100 x 1
       uX = np.dot(inputLayer, allInputs[x])

       # Prev Hidden(feeds last hidden) to Hidden
       hH = np.dot(hiddenHidden, allHiddenStates[x-1]) + biasHidden
        
       # Computes current hidden state, and adds it to dict
       allHiddenStates[x] = np.tanh((uX + hH))

       # Output from hidden to output
       allOutputs[x] = np.dot(hiddenOutput, allHiddenStates[x]) + biasOutput

       # Probabilties for next char
       # expo of all outputs / sum of all expos
       allProbabilties[x] = (np.exp(allOutputs[x])/np.sum(np.exp(allOutputs[x])))

       # Checks prob value for correct target
       # adds as loss usng negative loss
       # Softmax and calculate loss
       totalLoss += -np.log(allProbabilties[x][targets[x]][0])

    # Backward pass
    # go backwards through timestamps above
    deltaHiddenOutput = np.zeros_like(hiddenOutput)
    deltaInputLayer = np.zeros_like(inputLayer)
    deltaHiddenHidden = np.zeros_like(hiddenHidden)
    deltaPrevHiddenBias = np.zeros_like(allHiddenStates[-1])

    deltaBiasOutput = np.zeros_like(biasOutput)
    deltaBiasHidden = np.zeros_like(biasHidden)
    for b in range(len(inputs)-1, 0, -1):

        # Sum up gradients for every timesteps

        # Creates copy to do math on output
        outputError = np.copy(allProbabilties[b])

        print(outputError)
        
        # Calculates error at that target position
        # error
        outputError[targets[b]] -= 1

        # gradient for output layer
        # apply gradient to output
        # dE/dO = (o - t) * hiddenCurrentOutput
        deltaHiddenOutput += np.dot(outputError, allHiddenStates[b].T)
        deltaBiasOutput = deltaBiasOutput + outputError

        # backprop into H with previous hidden state graident bias
        # Basically new error for layers after output backprop
        # spread error to hidden layers, make it compatable to hidden matrix shape
        # also add bias from previous layer to error
        errorHidden = np.dot(hiddenOutput.T, outputError) + deltaPrevHiddenBias
      
        # error x tanh derivative
        deltaHiddenRaw = errorHidden * (1 - allHiddenStates[b]**2)
        deltaBiasHidden = deltaBiasHidden + deltaHiddenRaw

        # derivative for hidden layers
        # apply gradient to hidden
        # dE/dH = (1 - tanh(x)^2) * (Hidden-1) * dE/dO -> (Carry error back from output)
        # deltaHiddenRaw = dE/dO * (1 - tanh(x)^2)
        deltaHiddenHidden += np.dot(deltaHiddenRaw, allHiddenStates[b-1].T)

        # derivative for input weights
        # apply gradient to input
        # dE/dU = X * (1 - tanh(x)^2) * dE/dO
        # deltaHiddenRaw = dE/dO * (1 - tanh(x)^2)
        deltaInputLayer += np.dot(deltaHiddenRaw, allInputs[b].T)

        # Carry gradient for next timestep backstep
        # save graident to use in error for next hidden state
        # pass current hidden gradient to next hidden state
        deltaPrevHiddenBias = np.dot(hiddenHidden.T, deltaHiddenRaw)

    return {
        "loss": totalLoss
        "deltaHiddenOutput": deltaHiddenOutput,
        "deltaHiddenHidden": deltaHiddenHidden,
        "deltaInputLayer": deltaInputLayer,
        "deltaBiasOutput": deltaBiasOutput,
        "deltaBiasHidden": deltaBiasHidden
    }



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

    # feed data into model
    modelRes = train(inputs, targets, tempMemory)

    # Adjust weights
    inputLayer += modelRes["deltaInputLayer"]
    hiddenHidden += modelRes["deltaHiddenHidden"]

    # Move up data pointer and iteration
    inputPosition += seqLength
    currentIteration += 1

    break


