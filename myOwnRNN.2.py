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

X_val = []
Y_val = []

for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i:i+seq_len])
    Y_val.append(sin_wave[i+seq_len])
    
X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)

learning_rate = 0.0001    
nepoch = 25               
inputSize = 50                   # length of sequence
hidden_dim = 100         
output_dim = 1

bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

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
        
        dU_t = np.zeros(U.shape)
        dV_t = np.zeros(V.shape)
        dW_t = np.zeros(W.shape)
        
        dU_timeStep = np.zeros(U.shape)
        dW_timeStep = np.zeros(W.shape)
        
        # forward pass
        for t in range(inputSize):

            newInput = np.zeros(x.shape)
            newInput[t] = x[t]
            
            inputU = np.dot(U, newInput)
            hiddenW = np.dot(W, previousState)
            
            inputPlusHidden = hiddenW + inputU
            s = sigmoid(inputPlusHidden)
            
            outputV = np.dot(V, s)
            
            timeState.append({'currentState':s, 'previousState':previousState})

            print(s.shape)
            exit()
            
            previousState = s

        # derivative of pred
        # error = (outputV - y)
        error = -(y - outputV) # left side of derivative
        
        # backward pass
        for t in range(inputSize):

            newInput = np.zeros(x.shape)
            newInput[t] = x[t]

            # Calculate output layer gradient
            # dE/dV = -2(t - o) * W
            dV_t = np.dot(error, timeState[t]['currentState'].T)

            for i in range(t-1, max(-1, t-bptt_truncate-1), -1):
                
                # Current hidden state x Previous Hidden State = 100 x 1
                # Current Hidden State = W * prevW
                # dCurrent/dprev
                # d/dprev = W*prev = W
                dW_timeStep = np.dot(W, timeState[t]['previousState'])
                # Input Layers x Input = 100 x 1
                dU_timeStep = np.dot(U, newInput)
                
                dW_t += dW_timeStep
                dU_t += dU_timeStep
                
            dV += dV_t
            dU += dU_t
            dW += dW_t

            if dU.max() > max_clip_value:
                dU[dU > max_clip_value] = max_clip_value
            if dV.max() > max_clip_value:
                dV[dV > max_clip_value] = max_clip_value
            if dW.max() > max_clip_value:
                dW[dW > max_clip_value] = max_clip_value
                
            
            if dU.min() < min_clip_value:
                dU[dU < min_clip_value] = min_clip_value
            if dV.min() < min_clip_value:
                dV[dV < min_clip_value] = min_clip_value
            if dW.min() < min_clip_value:
                dW[dW < min_clip_value] = min_clip_value
        
        # update
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

preds = []
for i in range(Y_val.shape[0]):
    x, y = X_val[i], Y_val[i]
    previousState = np.zeros((hidden_dim, 1))
    # For each time step...
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
plt.plot(Y_val[:, 0], 'r')
plt.show()


