from RNNDataParse import RNNDataHandle
from RNN import RNN


textData = [
    'Drones are agile things, but they’re not exactly known for their quick reactions.', 
    'If you want to knock one out of the sky, a well-thrown ball or even a spear should do the trick.', 
    'Not for much longer, though, as researchers from the University of Zurich have created a drone that can autonomously dodge objects thrown at it — even at close range.',
    'You can see the quadcopter showing off these skills in the video above (though no-one tested it with a wrench).',
    'And okay, some of those throws are pretty easy, but the drone is still reacting completely autonomously.', 
    'And although we’ve seen quadcopters that can maneuver around static objects like trees, avoiding moving items mid-air is much trickier.',
    'Giving drones an auto-dodge feature would be handy for a lot of use-cases.', 
    'It would make drones safer, allowing them to dodge flying birds or nearby humans.', 
    'It would also be helpful for military and law enforcement deployments.', 
    'If you have a drone monitoring a protest, for example, being able to dodge thrown objects is a very useful skill.',
    'Falanga says that dodging dynamic objects is beyond the ken of even the most commercial advanced drones on the market today.', 
    'He says Skydio’s R1 drone probably has the best autonomous features but “it still struggles with avoiding moving objects.”',
    'As Falanga and his colleagues, Suseong Kim and Davide Scaramuzza, unpack in their research paper, there are lots of reasons for this limitation.', 
    'Technical factors including the responsiveness of a drone’s motors and the latency of their sensors all create bottlenecks.', 
    'What’s easy for a human (well, most of the time) is incredibly tricky for electronics.',
    'The University of Zurich’s drone, though, has one big advantage over commercial quadcopters: an advanced sensor known as an event camera.', 
    'While traditional cameras record a set number of frames each second and pass them on to software for processing, event cameras only send data when the pixels in its field of vision change in intensity.', 
    'This means they use less data and have lower latency.', 
    'In other words: a quicker response time.',
    'Event cameras are still uncommon, though.', 
    'They cost thousands of dollars and are not usually seen outside of a research lab.', 
    'Falanga says they’ll eventually hit the mainstream, but it will take years of development to bring them down to a reasonable cost.', 
    '“Absolutely I think in the long run I think we’ll see more and more usage of these cameras,” he says.',
    'Until then, drones will remain vulnerable to anyone with a good eye and a strong throwing arm.'
    # "John ate Pizza with Jane.",
    # "Jane ate Waffles with Bob."
    # "John ate Cheese with Dog."
    # "Jane ate Pie with Bob."
]

rnnHand = RNNDataHandle(textData)
inputSize = rnnHand.buildVocab()
inputs, targets = rnnHand.buildInputTarget()
print(inputs)
print(targets)

hiddenPoss = int(inputSize*0.75)

net = RNN(learningRate=0.01, hiddenSize=hiddenPoss, decayRate=0.1)
net.addLayer(inputLayer=True, inputSize=inputSize)
net.addLayer()
net.addLayer()
net.addLayer()
# net.addLayer()
net.addLayer(outputLayer=True, outputSize=inputSize)
net.trainNetwork(inputs, targets, epochs=31)
net.saveNeuralNet("rnn_net")

lastOut = []

# Makes prediciton for next word
def makePrediction(wordInput):
    return net.predict(wordInput)

for i in range(0,31):
    # print(lastOut)

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
