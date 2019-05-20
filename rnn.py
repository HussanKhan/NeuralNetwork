import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

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
previousHiddenState = np.zeros((hiddenSize, outputSize))


# How far to backprop
backPropLimit = 5
learningRate = 0.001

# Stops vanishing gradient
def clipGrad(grad):
    if grad.max() > 10:
        grad[grad.max() > 10] =  10
    if grad.min() < -10:
        grad[grad.min() < -10] =  -10
    return grad
    

# Feed forward through network

# for every output
for epoch in tqdm(range(25)):
    for o in range(outputSine.shape[0]):

        # stores all timesteps
        timeSteps = []
        
        # Current input and output set
        currentInput = inputSine[o]
        target = outputSine[o]
        
        # Give RNN one input at a time
        for index in range(inputSize):
        
            # Normal input [1, 2, 5, 6]
            # Send to RNN - [0, 0, 5, 6]
            inputX = np.zeros((inputSize, 1))
            inputX[index] = currentInput[index]

            # inputX = np.array(inputX, ndmin=1)

            # print("Input")
            # print(inputX.shape)
            
            # Input to Hidden
            # 100 x 50 * 50 * 1 = 100 x 1
            xU = np.dot(iH, inputX)

            # print("Weights x Input")
            # print(xU.shape)
    
            # Hidden to Hidden
            # 100 x 100 * 100 x 1 = 100 x 1
            hW = np.dot(hH, previousHiddenState)

            # print("Hidden State x PrevHidden State")
            # print(hW.shape)

            # Current Hidden State
            # 100 x 1 + 100 x 1 = 100 x 1
            currentHiddenState = sigmoid(( xU + hW ))

            # print("CurrentHiddenState")
            # print(currentHiddenState.shape)
    
            # Output from current Hidden State
            # 1 x 100 * 100 x 1 = 1 x 1
            newOutput = np.dot(hO, currentHiddenState)
            # print("Output")
            # print(newOutput.shape)
    
            # Set previous hidden state
            # 100 x 1
            previousHiddenState = currentHiddenState
    
            # Add time step for later backprop
            timeSteps.append({ 'currentHiddenState': currentHiddenState, 'previousHiddenState': previousHiddenState, "xInput": inputX, "inputHidden": xU, "currentHiddenOutput": hW})
            
            # print({ 'currentHiddenState': currentHiddenState, 'previousHiddenState': previousHiddenState})

    
        # print(error.shape)
        # print(hO.shape)
        # print(hH.shape)
        # print(iH.shape)
    
        d_hO = np.zeros((outputSize, hiddenSize))
        d_hH = np.zeros((hiddenSize, hiddenSize))
        d_iH = np.zeros((hiddenSize, inputSize))

        # Calculate error of last output
        # 1x1 - 1x1 = 1x1
        error = (newOutput - target)

        # Go through times steps backwards
        currentTimeStep = inputSize-1
        for t in range(currentTimeStep, (currentTimeStep - backPropLimit), -1):
            
            # 1 x 1 * 100 x 1 = 1 x 100
            gradient = np.dot((error * (newOutput * (1 - newOutput))) , timeSteps[t]['currentHiddenState'].T)
            d_hO = d_hO + ((learningRate) * clipGrad(gradient))
    
            # Spread error to next layer
            # 1 x 100 * 1 x 1 = 100 x 1
            error = np.dot(hO.T, error) 

            # 100 x 1 * 100 x 1 = 100 x 100
            gradient = np.dot((error * (timeSteps[t]['currentHiddenState'] * (1 - timeSteps[t]['currentHiddenState']))) , timeSteps[t]['inputHidden'].T)
            d_hH = d_hH + ((learningRate) * clipGrad(gradient))
    
            # Spread error to next layer
            # 100 x 100 * 100 x 1 = 100 x 1
            error = np.dot(hH.T, error)

            # 100 x 1 * 50 x 1 = 100 x 50
            gradient = np.dot((error * (timeSteps[t]['inputHidden'] * (1 - timeSteps[t]['inputHidden']))) , timeSteps[t]['xInput'].T)
            
            d_iH = d_iH + ((learningRate) * clipGrad(gradient))
            
            # 100 x 1 * 100 x 1 =  1 x 1
            error = np.dot(timeSteps[t]['previousHiddenState'].T, error)
        
        # Update weights
        hO = hO + d_hO
        hH = hH + d_hH
        iH = iH + d_iH

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

print(preds[0])

preds = preds

plt.plot(preds[:, 0, 0], 'b')
plt.plot(outputSine[:], 'g')
plt.show()



        










