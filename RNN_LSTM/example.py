from RNNDataParse import RNNDataHandle
from RNN import RNN


textData = [
    "John went to space with Jane.",
    "Jane went to space with John.",
    "Dog went to space with Bob."
]

rnnHand = RNNDataHandle(textData)
inputSize = rnnHand.buildVocab()
inputs, targets = rnnHand.buildInputTarget()
print(inputs)
print(targets)

# hiddenNodes = int(inputSize*0.5)
hiddenNodes = inputSize

net = RNN(learningRate=0.3, hiddenSize=hiddenNodes)
net.addLayer(inputLayer=True, inputSize=inputSize)
net.addLayer(outputLayer=True, outputSize=inputSize)
net.trainNetwork(inputs, targets, epochs=4)
net.saveNeuralNet("rnn_net")

lastOut = []

# Makes prediciton for next word
def makePrediction(wordInput):
    return net.predict(wordInput)

for i in range(0,5):

    # If there is a prev prediction, feed it into network
    # and set the last prediction
    if len(lastOut):
        pred = makePrediction(lastOut)
        lastOut = pred.T
    else:
        # If no prev prediction make one
        pred = makePrediction(inputs[i])
        lastOut = pred.T
        print(rnnHand.getString(inputs[i]))

    print(rnnHand.getString(pred))
