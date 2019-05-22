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

# make sinWaveData into numpy array
sinWaveData = np.array(sinWaveData)
# print(sinWaveData)

# # Input Data structure
# # Given the first 50 numbers of sine, predict the 51st number
inputSize = 50

X = []
Y = []

for i in range(100):
    X.append(sinWaveData[i:i+inputSize])
    Y.append(sinWaveData[i+inputSize])
    
X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)

hiddenSize = 100
outputSize = 1

# Input to Hidden
iH = np.random.uniform(0, 1, (hiddenSize, inputSize)) 
# Previous Hiden State to Current Hidden State
hH = np.random.uniform(0, 1, (hiddenSize, hiddenSize))
# Hidden State to output
hO = np.random.uniform(0, 1, (outputSize, hiddenSize))

# How far to backprop
backPropLimit = 5
lrDecayed = lambda x: (1 / ((1 + 0.1) * x)) * 0.001
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

    #learningRate = lrDecayed(epoch+1)

    # Go over output one at a time
    for o in range(Y.shape[0]):

        # Current input and output set
        currentInput, target = X[o], Y[o]
        
        # stores all timesteps
        timeSteps = []
        # Initial Hidden is just zeros
        previousHiddenState = np.zeros((hiddenSize, outputSize))

        # Chaning in main weights
        dU = np.zeros(iH.shape)
        dV = np.zeros(hO.shape)
        dW = np.zeros(hH.shape)
        # change in timestep weight
        dU_t = np.zeros(iH.shape)
        dV_t = np.zeros(hO.shape)
        dW_t = np.zeros(hH.shape)
        # changine in input
        dU_i = np.zeros(iH.shape)
        dW_i = np.zeros(hH.shape)
        
        # Give RNN one input at a time
        for index in range(inputSize):
        
            inputX = np.zeros(currentInput.shape)
            inputX[index] = currentInput[index]
            
            xU = (np.dot(iH, inputX))
            lastInput = xU
    
            hW = (np.dot(hH, previousHiddenState))
            lastWeight = hW

            lastHiddenOutput = ( xU + hW )
            currentHiddenState = sigmoid(lastHiddenOutput)
            newOutput = (np.dot(hO, currentHiddenState))

            timeSteps.append({ 'currentHiddenState': currentHiddenState, 'previousHiddenState': previousHiddenState, "xInput": inputX, "inputHidden": xU, "currentHiddenOutput": hW, "output": newOutput})

            previousHiddenState = currentHiddenState

        # derivative of pred
        error = (newOutput - target)

        print("\r" + str(error), end="", flush=True)
        
        # backward pass
        for t in range(inputSize):

            # Change is output weights
            dV_t = np.dot(error, timeSteps[t]["currentHiddenState"].T)

            # Spread error to next layer
            nextLayerError = np.dot(hO.T, error)
            
            ds = nextLayerError

            dadd = lastHiddenOutput * (1 - lastHiddenOutput) * ds
            
            dWeights = dadd * np.ones_like(lastWeight)

            # Spread change to Hidden Weights
            dPrevState = np.dot(hH.T, dWeights)


            for i in range(t-1, max(-1, t-backPropLimit-1), -1):
                
                ds = nextLayerError + dPrevState

                dHiddenState = lastHiddenOutput * (1 - lastHiddenOutput) * ds

                dWeights = dHiddenState * np.ones_like(lastWeight)
                dInputLayer = dHiddenState * np.ones_like(lastInput)

                dW_i = np.dot(hH, timeSteps[t]["previousHiddenState"])
                
                dPrevState = np.dot(hH.T, dWeights)

                new_input = np.zeros(currentInput.shape)
                new_input[t] = currentInput[t]

                dU_i = np.dot(iH, new_input)
                dx = np.dot(iH.T, dInputLayer)

                dU_t += dU_i
                dW_t += dW_i
                
            dV += dV_t
            dU += dU_t
            dW += dW_t

            dU = clipGrad(dU)
            dV = clipGrad(dV)
            dW = clipGrad(dW)

        # update
        iH -= learningRate * dU
        hO -= learningRate * dV
        hH -= learningRate * dW

preds = []
for i in range(Y.shape[0]):
    x, y = X[i], Y[i]
    prev_s = np.zeros((hiddenSize, 1))
    # Forward pass
    for t in range(inputSize):
        mulu = np.dot(iH, x)
        mulw = np.dot(hH, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(hO, s)
        prev_s = s

    preds.append(mulv)
    
preds = np.array(preds)

plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y[:, 0], 'r')
plt.show()









