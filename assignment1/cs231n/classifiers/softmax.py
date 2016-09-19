import numpy as np
from random import shuffle

def softmax(l):
  if len(l.shape) == 1:
    l -= l.max()
    return np.exp(l) / np.sum(np.exp(l))
  else:
    l -= np.max(l, axis=1)[:, np.newaxis]
    return np.exp(l) / np.sum(np.exp(l), axis=1)[:, np.newaxis]

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  loss = 0.0
  for i in xrange(num_train):
    probs = softmax(X[i].dot(W))
    loss -= np.log(probs)[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += X[i] * (probs[j] - 1)
      else:
        dW[:, j] += X[i] * (probs[j])

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  probs = softmax(np.dot(X, W))
  loss -= np.sum(np.log(probs[np.arange(num_train), y]))
  probs[np.arange(num_train), y] -= 1
  dW += np.dot(X.T, probs)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW

