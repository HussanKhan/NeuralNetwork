import numpy as np
import random

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
learningRate = 1e-1

# Model parameters - RNN shape
np.random.seed(4)
inputLayer = np.random.randn(hiddenSize, vocabSize)*0.01
np.random.seed(7)
hiddenHidden = np.random.randn(hiddenSize, hiddenSize)*0.01
np.random.seed(10)
hiddenOutput = np.random.randn(vocabSize, hiddenSize)*0.01

biasHidden = np.zeros((hiddenSize, 1)) # hidden bias
biasOutput = np.zeros((vocabSize, 1)) # output bias

# Hold cumlative changes for layer learning rate change
minputLayer  = np.zeros_like(inputLayer)
mhiddenHidden = np.zeros_like(hiddenHidden)
mhiddenOutput = np.zeros_like(hiddenOutput)
mbiasHidden = np.zeros_like(biasHidden)
mbiasOutput = np.zeros_like(biasOutput)

############## Training ##############

# Clips exploding Gradient
def clipGradient(deltas):
    if deltas.max() > 5:
        deltas[deltas.max() > 5] =  5
    if deltas.min() < -5:
        deltas[deltas.min() < -5] =  -5
    return 0

def train(inputs, targets, prevHiddenStates):

    # Stores each timestep for later backprop
    allInputs = {}
    allHiddenStates = {-1: np.copy(prevHiddenStates)}
    allOutputs = {}
    allProbabilties = {}

    # Stores total loss for backprop
    totalLoss = 0

    # Forward pass
    for x in xrange(len(inputs)):
       # One-hot input
       allInputs[x] = np.zeros((vocabSize, 1))
       allInputs[x][inputs[x]] = 1
       
       #Feed input to layer
       #100 x 65 * 65 x 1 = 100 x 1
       uX = np.dot(inputLayer, allInputs[x])

       # Prev Hidden(feeds last hidden) to Hidden
       hH = np.dot(hiddenHidden, allHiddenStates[x-1])
        
       # Computes current hidden state, and adds it to dict
       allHiddenStates[x] = np.tanh(uX + hH + biasHidden)

       # Output from hidden to output
       allOutputs[x] = np.dot(hiddenOutput, allHiddenStates[x]) + biasOutput
    
       # Probabilties for next char
       # expo of all outputs / sum of all expos
       allProbabilties[x] = (np.exp(allOutputs[x])/np.sum(np.exp(allOutputs[x])))

       # Checks prob value for correct target
       # adds as loss usng negative loss
       # Softmax and calculate loss
       totalLoss += -np.log(allProbabilties[x][targets[x],0]) # softmax (cross-entropy loss)

    # Backward pass
    # go backwards through timestamps above
    deltaHiddenOutput = np.zeros_like(hiddenOutput)
    deltaInputLayer = np.zeros_like(inputLayer)
    deltaHiddenHidden = np.zeros_like(hiddenHidden)
    deltaPrevHiddenBias = np.zeros_like(allHiddenStates[0])

    deltaBiasOutput = np.zeros_like(biasOutput)
    deltaBiasHidden = np.zeros_like(biasHidden)
    for b in range(len(inputs)-1, -1, -1):
        # Sum up gradients for every timesteps

        # Creates copy to do math on output
        outputError = np.copy(allProbabilties[b])
        
        # Calculates error at that target position
        # error
        outputError[targets[b]] -= 1

        # gradient for output layer
        # apply gradient to output
        # dE/dO = (o - t) * hiddenCurrentOutput
        deltaHiddenOutput += np.dot(outputError, allHiddenStates[b].T)
        deltaBiasOutput += outputError

        # backprop into H with previous hidden state graident bias
        # Basically new error for layers after output backprop
        # spread error to hidden layers, make it compatable to hidden matrix shape
        # also add bias from previous layer to error
        errorHidden = np.dot(hiddenOutput.T, outputError) + deltaPrevHiddenBias
      
        # error x tanh derivative
        deltaHiddenRaw = errorHidden * (1 - allHiddenStates[b]**2)
        deltaBiasHidden += deltaHiddenRaw

        # derivative for input weights
        # apply gradient to input
        # dE/dU = X * (1 - tanh(x)^2) * dE/dO
        # deltaHiddenRaw = dE/dO * (1 - tanh(x)^2)
        deltaInputLayer += np.dot(deltaHiddenRaw, allInputs[b].T)

        # derivative for hidden layers
        # apply gradient to hidden
        # dE/dH = (1 - tanh(x)^2) * (Hidden-1) * dE/dO -> (Carry error back from output)
        # deltaHiddenRaw = dE/dO * (1 - tanh(x)^2)
        deltaHiddenHidden += np.dot(deltaHiddenRaw, allHiddenStates[b-1].T)

        # Carry gradient for next timestep backstep
        # save graident to use in error for next hidden state
        # pass current hidden gradient to next hidden state
        deltaPrevHiddenBias = np.dot(hiddenHidden.T, deltaHiddenRaw)

    # Clips Gradients
    deltasArr = [deltaBiasOutput, deltaBiasHidden, deltaHiddenOutput, deltaHiddenHidden, deltaInputLayer]

    for d in deltasArr:
        np.clip(d, -5, 5, out=d)
        # clipGradient(d)

    return {
        "loss": totalLoss,
        "deltaHiddenOutput": deltaHiddenOutput,
        "deltaHiddenHidden": deltaHiddenHidden,
        "deltaInputLayer": deltaInputLayer,
        "deltaBiasOutput": deltaBiasOutput,
        "deltaBiasHidden": deltaBiasHidden,
        "lastHiddenState": allHiddenStates[len(inputs) - 1] #last hidden state
    }

############## Predict ##############
def predict(seed, n, prevState):
    
    # Create first input
    xInput = np.zeros((vocabSize, 1))
    xInput[seed] = 1

    # Stores all predicited chars
    predictedChars = []

    # Normal feed forward
    for i in range(n):
        
        uX = np.dot(inputLayer, xInput)
        hH = np.dot(hiddenHidden, prevState)
        
        currentHiddenState = np.tanh(hH + uX + biasHidden)
        
        output = np.dot(hiddenOutput, currentHiddenState) + biasOutput
        
        # Softmax output
        output = np.exp(output) / np.sum(np.exp(output))

        # Random choice between using distribution from softmax above
        bestGuess = np.random.choice(range(vocabSize), p=output.ravel())
        predictedChars.append(bestGuess)
        
        # Makes best guess into new input
        xInput = np.zeros((vocabSize, 1))
        xInput[bestGuess] = 1

    # Plain next
    plainText = []
    for index in predictedChars:
        plainText.append(indexToChar[index])

    return "".join(plainText)


inputPosition = 0
currentIteration = 0

smooth_loss = -np.log(1.0/vocabSize)*seqLength # loss at iteration 0

# Keep training until user stops
while True:

    # If end of input is reached, reset
    if inputPosition+seqLength+1 >= dataSize or currentIteration == 0:
        # Resets RNN temp mem
        # Hidden states are only kept for one target at a time
        # as all those n inputs lead to that target
        tempMemory = np.zeros((hiddenSize, 1)) # Memory from all inputs for one target
        inputPosition = 0 # go back and start new epoch

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

    # feed data into model
    modelRes = train(inputs, targets, tempMemory)

    # set new hidden state, so network keeps track of time
    tempMemory = modelRes["lastHiddenState"]
    
    # Constally calculate loss
    smooth_loss = smooth_loss * 0.999 + modelRes["loss"] * 0.001

    # Make prediction, check status
    if currentIteration % 100 == 0:
        pred = predict(inputs[0], 200, tempMemory)
        print(pred)
        print("Loss: {} Iter: {}".format(smooth_loss, currentIteration))

    # Adjust weights
    weigthArr = [inputLayer, hiddenHidden, hiddenOutput, biasOutput, biasHidden]
    deltaArr = [modelRes["deltaInputLayer"], modelRes["deltaHiddenHidden"], modelRes["deltaHiddenOutput"], modelRes["deltaBiasOutput"], modelRes["deltaBiasHidden"]]
    deltaMem = [minputLayer, mhiddenHidden, mhiddenOutput, mbiasOutput, mbiasHidden]

    for weight, deltaWeight, deltaMem in zip(weigthArr, deltaArr, deltaMem):
        
        deltaMem += deltaWeight * deltaWeight
        weight += -learningRate * (deltaWeight / np.sqrt(deltaMem + 1e-8)) # slowly update weights based on past changes

    # Move up data pointer and iteration
    inputPosition += seqLength
    currentIteration += 1

