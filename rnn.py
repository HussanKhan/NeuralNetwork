import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def tanh(x):
    return np.tanh(x)

sinWaveData = []
# Fill sine data at position x
for s in range(200):
    sinWaveData.append(math.sin(s))

# make sineWaveData into numpy array
sinWaveData = np.array(sinWaveData)
# print(sinWaveData)

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
backPropLimit = 15
# lrDecayed = lambda x: (1 / ((1 + 0.1) * x)) * 0.01
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
print("\n")
for epoch in range(25):

    # learningRate = lrDecayed(epoch+1)
    
    # Go over output one at a time
    for o in range(outputSine.shape[0]):

        # sum of updates
        d_hO = np.zeros((outputSize, hiddenSize))
        d_hH = np.zeros((hiddenSize, hiddenSize))
        d_iH = np.zeros((hiddenSize, inputSize))

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
            
            # Input to Hidden
            # 100 x 50 * 50 * 1 = 100 x 1
            xU = (np.dot(iH, inputX))
    
            # Hidden to Hidden
            # 100 x 100 * 100 x 1 = 100 x 1
            hW = (np.dot(hH, previousHiddenState))

            # Current Hidden State
            # 100 x 1 + 100 x 1 = 100 x 1
            currentHiddenState = tanh(( xU + hW ))

            # Output from current Hidden State
            # 1 x 100 * 100 x 1 = 1 x 1
            newOutput = (np.dot(hO, currentHiddenState))
    
            # Set previous hidden state
            # 100 x 1
            previousHiddenState = currentHiddenState
    
            # Add time step for later backprop
            timeSteps.append({ 'currentHiddenState': currentHiddenState, 'previousHiddenState': previousHiddenState, "xInput": inputX, "inputHidden": xU, "currentHiddenOutput": hW, "output": newOutput})

        # Go through times steps backwards
        currentTimeStep = inputSize-1

        for step in range(inputSize):

            error = (target - timeSteps[currentTimeStep-step]["output"])

            # 1 x 1 * 100 x 1 = 1 x 100
            # gradient = (learningRate * gradient)
            gradientO = np.dot((error * (1 - timeSteps[currentTimeStep-step]["output"]**2)), timeSteps[currentTimeStep-step]['previousHiddenState'].T)
            
            print("\r" + str(gradientO), end="", flush=True)

            for t in range(currentTimeStep-step, (step - backPropLimit), -1):
                # Spread error to next layer
                # 1 x 100 * 1 x 1 = 100 x 1
                # error = np.dot(hO.T, error)
                # # error = (error) * learningRate

                # # 100 x 1 * 100 x 1 = 100 x 100
                # gradientH = np.dot((error * ((1 - timeSteps[t]['currentHiddenState']**2))) , (timeSteps[t]['previousHiddenState'] + timeSteps[t]['inputHidden']).T)
                # gradient = (learningRate * gradient)

                # 1 x 100 * 1 x 100 =  100 x 100
                gradientH = np.dot(hO.T, gradientO)
                
                # Spread error to next layer
                # 100 x 100 * 100 x 1 = 100 x 1
                # error = np.dot(hH.T, error)
                # error = (error) * learningRate

                # 100 x 1 * 50 x 1 = 100 x 50
                # gradientI = np.dot((error * ((1 - timeSteps[t]['inputHidden']**2))) , timeSteps[t]['xInput'].T)

                # 100 x 100 * 100 * 100
                gradientI = np.dot(hH.T, gradientH)
                # gradientI = np.dot(hH.T, gradientH)
                # gradient = (learningRate * gradient)

                # 100 x 1 * 100 x 1 =  1 x 1
                # error = np.dot(timeSteps[t]['previousHiddenState'].T, error)
                # error = (error) * learningRate

            d_hO = d_hO + learningRate * (clipGrad(gradientO))
            d_hH = d_hH + learningRate * (clipGrad(gradientH))
            # d_iH = d_iH + learningRate * (clipGrad(gradientI))

        # Update weights
        hO = hO + (learningRate *d_hO)
        hH = hH + (learningRate *d_hH)
        iH = iH + (learningRate *d_iH)

preds = []
print(hO)
print(hH)
print(iH)
for i in range(outputSine.shape[0]):
    currentInput = inputSine[i]
    target = outputSine[i]
    
    prev_s = np.zeros((hiddenSize, 1))
    
    # Forward pass
    for t in range(inputSize):

        xH = (np.dot(iH, currentInput))
        wH = (np.dot(hH, prev_s))
        s = tanh(xH + hH)
        mulv = np.dot(hO, s)
        prev_s = s

    preds.append(mulv)
    
preds = np.array(preds)

print(preds[:,0,0])
print(outputSine[:])

preds = preds

plt.plot(preds[:,0,0], 'b')
plt.plot(outputSine[:], 'g')
plt.show()



        










