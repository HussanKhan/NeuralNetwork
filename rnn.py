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
backPropLimit = 5
lrDecayed = lambda x: (1 / ((1 + 0.2) * x)) * 0.001
learningRate = 0

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
    learningRate = lrDecayed(epoch+1)
    # print(learningRate)
    for o in range(outputSine.shape[0]):

        # stores all timesteps
        timeSteps = []
        
        # Current input and output set
        currentInput = inputSine[o]
        target = outputSine[o]
        # print("TEST")
        # print(target)
        
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
            xU = tanh(np.dot(iH, inputX))

            # print("Weights x Input")
            # print(xU.shape)
    
            # Hidden to Hidden
            # 100 x 100 * 100 x 1 = 100 x 1
            hW = tanh(np.dot(hH, previousHiddenState))

            # print("Hidden State x PrevHidden State")
            # print(hW.shape)

            # Current Hidden State
            # 100 x 1 + 100 x 1 = 100 x 1
            currentHiddenState = tanh(( xU + hW ))

            # print(currentHiddenState)

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
        # print("NEW ERROR")
        error = (target - newOutput)
        print("\r" + str(error), end="", flush=True)
        # print("out")
        # print(newOutput)
        # print("tar")
        # print(target)
        # Go through times steps backwards
        currentTimeStep = inputSize-1
        for t in range(currentTimeStep, (currentTimeStep - backPropLimit), -1):
            # print(error)
            
            # 1 x 1 * 100 x 1 = 1 x 100
            gradient = tanh(np.dot((error * ((1 - newOutput**2))) , timeSteps[t]['currentHiddenState'].T))
            # print("GRADIENT hO") 
            # print(gradient)
            # print(gradient.max())
            # print(error * (1 - newOutput**2))
            # print(timeSteps[t]['currentHiddenState'])
            d_hO = tanh(d_hO + clipGrad(gradient))
    
            # Spread error to next layer
            # 1 x 100 * 1 x 1 = 100 x 1
            error = tanh(np.dot(hO.T, error))
            
            # print("ERROR")
            # print(error)

            # 100 x 1 * 100 x 1 = 100 x 100
            gradient = tanh(np.dot((error * ((1 - timeSteps[t]['currentHiddenState']**2))) , (timeSteps[t]['previousHiddenState'] + timeSteps[t]['inputHidden']).T))
            # print("GRADIENT hH")
            # print(gradient)
            d_hH = tanh(d_hH + clipGrad(gradient))
    
            # Spread error to next layer
            # 100 x 100 * 100 x 1 = 100 x 1
            error = tanh(np.dot(hH.T, error))
            
            # print("HH")
            # print(hH)
            # print("ERROR") 
            # print(error)

            # 100 x 1 * 50 x 1 = 100 x 50
            gradient = tanh(np.dot((error * ((1 - timeSteps[t]['inputHidden']**2))) , timeSteps[t]['xInput'].T))
            # print("GRADIENT iH")
            # print(gradient)
            
            d_iH = tanh(d_iH + clipGrad(gradient))
            
            # 100 x 1 * 100 x 1 =  1 x 1
            error = tanh(np.dot(timeSteps[t]['previousHiddenState'].T, error))
            
            # print("ERROR") 
            # print(error)
            # exit()
        
        # Update weights
        hO = tanh(hO + (learningRate * d_hO))
        hH = tanh(hH + (learningRate * d_hH))
        iH = tanh(iH + (learningRate * d_iH))
        # print(d_hO)
        # print(hH)
        # print(iH)

preds = []
for i in range(outputSine.shape[0]):
    currentInput = inputSine[i]
    target = outputSine[i]
    
    prev_s = np.zeros((hiddenSize, 1))
    
    # Forward pass
    for t in range(inputSize):

        xH = tanh(np.dot(iH, currentInput))
        wH = tanh(np.dot(hH, prev_s))
        s = tanh(xH + hH)
        mulv = tanh(np.dot(hO, s))
        prev_s = s

    preds.append(mulv)
    
preds = np.array(preds)

print(preds[:,0,0])
print(outputSine[:])

preds = preds

plt.plot(preds[:,0,0], 'b')
plt.plot(outputSine[:], 'g')
plt.show()



        










