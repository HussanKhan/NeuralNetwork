import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
learningRate = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize neural network weights
inputLayer = 2*np.random.random((input_dim,hidden_dim)) - 1
outputLayer = 2*np.random.random((hidden_dim,output_dim)) - 1
hiddenLayer = 2*np.random.random((hidden_dim,hidden_dim)) - 1

inputLayer_update = np.zeros_like(inputLayer)
outputLayer_update = np.zeros_like(outputLayer)
hiddenLayer_update = np.zeros_like(hiddenLayer)

# training logic
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,inputLayer) + np.dot(layer_1_values[-1],hiddenLayer))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,outputLayer))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(hiddenLayer.T) + layer_2_delta.dot(outputLayer.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        outputLayer_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        hiddenLayer_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        inputLayer_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    inputLayer += inputLayer_update * learningRate
    outputLayer += outputLayer_update * learningRate
    hiddenLayer += hiddenLayer_update * learningRate    

    inputLayer_update *= 0
    outputLayer_update *= 0
    hiddenLayer_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"