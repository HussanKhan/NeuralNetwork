"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(inputLayer, xs[t]) + np.dot(hiddenHidden, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(hiddenOutput, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  
  # backward pass: compute gradients going backwards
  dinputLayer, dhiddenHidden, dhiddenOutput = np.zeros_like(inputLayer), np.zeros_like(hiddenHidden), np.zeros_like(hiddenOutput)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  
  for t in reversed(xrange(len(inputs))):

    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

    dhiddenOutput += np.dot(dy, hs[t].T)
    dby += dy
    
    dh = np.dot(hiddenOutput.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    
    dinputLayer += np.dot(dhraw, xs[t].T)
    dhiddenHidden += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(hiddenHidden.T, dhraw)
  
  for dparam in [dinputLayer, dhiddenHidden, dhiddenOutput, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  
  return loss, dinputLayer, dhiddenHidden, dhiddenOutput, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(inputLayer, x) + np.dot(hiddenHidden, h) + bh)
    y = np.dot(hiddenOutput, h) + by

    p = np.exp(y) / np.sum(np.exp(y))
    
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    print(p)
    print(ix)
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes


# data I/O
# data = open('input.txt', 'r').read() # should be simple plain text file
data = open('prideP.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 500 # size of hidden layer of neurons
seq_length = 40 # number of steps to unroll the RNN for
# learning_rate = 1e-1
learning_rate = 0.001


# model parameters
inputLayer = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
hiddenHidden = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
hiddenOutput = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output

bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

n, p = 0, 0

minputLayer, mhiddenHidden, mhiddenOutput = np.zeros_like(inputLayer), np.zeros_like(hiddenHidden), np.zeros_like(hiddenOutput)

mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

while True:
  
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data

  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dinputLayer, dhiddenHidden, dhiddenOutput, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([inputLayer, hiddenHidden, hiddenOutput, bh, by], 
                                [dinputLayer, dhiddenHidden, dhiddenOutput, dbh, dby], 
                                [minputLayer, mhiddenHidden, mhiddenOutput, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
