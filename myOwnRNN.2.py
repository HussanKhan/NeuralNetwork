import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

sin_wave = np.array([math.sin(x) for x in np.arange(200)])

X = []
Y = []

seq_len = 50
num_records = len(sin_wave) - seq_len

for i in range(num_records - 50):
    X.append(sin_wave[i:i+seq_len])
    Y.append(sin_wave[i+seq_len])

# 100 Records stack on top 3d
# 50 Rows 1 Column
X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)

learning_rate = 0.0001    
nepoch = 25               
inputSize = 50                   # length of sequence
hidden_dim = 100         
output_dim = 1

bptt_truncate = 5

# Stops vanishing gradient
def clipGrad(grad):
    if grad.max() > 10:
        grad[grad.max() > 10] =  10
    if grad.min() < -10:
        grad[grad.min() < -10] =  -10
    return 0

np.random.seed(4) # 4 9 15
U = np.random.uniform(0, 1, (hidden_dim, inputSize))
np.random.seed(7) # 7 8 14
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
np.random.seed(10) # 10 7 13
V = np.random.uniform(0, 1, (output_dim, hidden_dim))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in tqdm(range(nepoch)):

    # train model
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
    
        timeState = []
        previousState = np.zeros((hidden_dim, 1))

        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        dW = np.zeros(W.shape)
        
        # forward pass
        for t in range(inputSize):

            newInput = np.zeros(x.shape)
            newInput[t] = x[t]

            inputU = np.dot(U, newInput)
            hiddenW = np.dot(W, previousState)
            
            inputPlusHidden = hiddenW + inputU
            s = sigmoid(inputPlusHidden)
            
            outputV = np.dot(V, s)
            
            timeState.append({'currentState':s})

            previousState = s

        # derivative of pred
        # error = (outputV - y)
        error = -(y - outputV) # left side of derivative
        
        # backward pass
        for t in range((inputSize-1), -1,-1):

            newInput = np.zeros(x.shape)
            newInput[t] = x[t]

            # Add up gradients of output layer and input layer per output
            dV += np.dot(error, timeState[t]['currentState'].T)
            dU += np.dot(U, newInput)
            
            # Add Up gradient of previous 5 hidden states
            for i in range(t, max(-1, t-bptt_truncate), -1):
                dW += np.dot(W, timeState[i]['currentState'])

            # Clips very low or very large weight changes
            clipGrad(dV)
            clipGrad(dW)
            clipGrad(dU)

        # Do opposite of gradient
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW

preds = []
for i in range(Y.shape[0]):
    x, y = X[i], Y[i]
    previousState = np.zeros((hidden_dim, 1))
    # Forward pass
    for t in range(inputSize):
        inputU = np.dot(U, x)
        hiddenW = np.dot(W, previousState)
        inputPlusHidden = hiddenW + inputU
        s = sigmoid(inputPlusHidden)
        outputV = np.dot(V, s)
        previousState = s

    preds.append(outputV)
    
preds = np.array(preds)

plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y[:, 0], 'r')
plt.show()

