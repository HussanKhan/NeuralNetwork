import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sinWaveData = []
# Fill sine data at position x
for s in range(200):
    sinWaveData.append(math.sin(s))

# make sineWaveData into numpy array
sinWaveData = np.array(sinWaveData)
print(sinWaveData)

# create inputs and outputs
inputSine = []
outputSine = []

# Input Data structure
# Given the first 50 numbers of sine, predict the 51st number
inputSize = 50
for i in range(50):

    # First 50
    inputSection = sinWaveData[i : i + inputSize]
    # 51st, predict this v
    output = sinWaveData[i + inputSize]

    inputSine.append(inputSection)
    outputSine.append(output)

inputSine = np.array(inputSine)
outputSine = np.array(outputSine)

hiddenSize = 100
outputSize = 1

# Input to Hidden
iH = np.random.uniform(0, 1, (hiddenSize, inputSize)) 
# Previous Hiden State to Current Hidden State
hH = np.random.uniform(0, 1, (hiddenSize, hiddenSize))
# Hidden State to output
hO = np.random.uniform(0, 1, (outputSize, hiddenSize))

# Initial Hidden is just zeros
previousHiddenState = np.zeros((outputSize, hiddenSize))


# How far to backprop
backPropLimit = 10
learningRate = 0.0001

# Stops vanishing gradient
def clipGrad(grad):
    if grad.max() > 10:
        grad[grad.max() > 10] =  10
    if grad.min() < -10:
        grad[grad.min() < -10] =  -10
    return grad
    

# Feed forward through network

# for every output
for epoch in range(25):
    for o in range(outputSine.shape[0]):

        # stores all timesteps
        timeSteps = []
        
        print(o)
        
        # Current input and output set
        currentInput = inputSine[o]
        target = outputSine[o]
        
        # Give RNN one input at a time
        for index in range(inputSize):
        
            # Normal input [1, 2, 5, 6]
            # Send to RNN - [0, 0, 5, 6]
            inputX = np.zeros(currentInput.shape)
            inputX[index] = currentInput[index]
            
            # Input to Hidden
            xU = np.dot(iH, inputX)
            
            # Hidden to Hidden
            hW = np.dot(hH, previousHiddenState.T)
    
            # Current Hidden State
            currentHiddenState = sigmoid(( xU + hW ))
    
            # Output from current Hidden State
            newOutput = np.dot(hO, currentHiddenState)
    
            # Set previous hidden state
            previousHiddenState = currentHiddenState
    
            # Add time step for later backprop
            timeSteps.append({ 'currentHiddenState': currentHiddenState, 'previousHiddenState': previousHiddenState, "xInput": inputX, "inputHidden": xU})
            
            # print({ 'currentHiddenState': currentHiddenState, 'previousHiddenState': previousHiddenState})
    
        # Calculate error of last output
        error = (newOutput - target)
    
        print(error.shape)
        print(hO.shape)
        print(hH.shape)
        print(iH.shape)
    
    
        # Go through times steps backwards
        currentTimeStep = inputSize-1
        for t in range(currentTimeStep, (currentTimeStep - backPropLimit), -1):
        
            gradient = np.dot((error * (newOutput * (1 - newOutput))) , timeSteps[t]['currentHiddenState'])
    
            hO = hO + (learningRate) * clipGrad(gradient)
    
            # Spread error to next layer
            error = np.dot(hO.T, error) 
            gradient = np.dot((error * (timeSteps[t]['currentHiddenState'] * (1 - timeSteps[t]['currentHiddenState']))) , timeSteps[t]["previousHiddenState"])
            hH = hH + (learningRate) * clipGrad(gradient)
    
            # Spread error to next layer
            error = np.dot(hH.T, error)
            gradient = np.dot((error * (timeSteps[t]['inputHidden'] * (1 - timeSteps[t]['inputHidden']))) , iH)
            iH = iH + (learningRate) * clipGrad(gradient)
    
            print(iH)

preds = []
for i in range(outputSine.shape[0]):
    currentInput = inputSine[i]
    target = outputSine[i]
    
    prev_s = np.zeros((hiddenSize, 1))
    
    # Forward pass
    for t in range(inputSize):

        xH = np.dot(iH, currentInput)
        wH = np.dot(hH, prev_s)

        s = sigmoid(xH + hH)

        mulv = np.dot(hO, s)
        
        prev_s = s

    preds.append(mulv)
    
preds = np.array(preds)

print(preds)

preds = preds

plt.plot(preds[:, 0, 0], 'g')
plt.plot(outputSine[:], 'r')
plt.show()



        










