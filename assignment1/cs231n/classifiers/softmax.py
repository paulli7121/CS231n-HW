import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = X.dot(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim = W.shape[0]
  for sample in range(num_train):
    f = scores[sample]
    f -= np.max(f)
    p = np.exp(f[y[sample]]) / np.sum(np.exp(f))
    loss += -np.log(p)
    for cla in range(num_class):
        dW[:,cla] += np.exp(f[cla])/np.sum(np.exp(f)) *X[sample]
    dW[:,y[sample]] -= X[sample]
  loss = loss/num_train
  loss += reg * np.sum(W * W)
  dW = dW/num_train
  dW += 2 * reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = X.dot(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim = W.shape[0]

  f = scores - np.max(scores)
  p = np.exp(f[range(num_train), y]) / np.sum(np.exp(f), axis = 1)
  loss = -np.sum(np.log(p))
  loss = loss/num_train
  loss += reg * np.sum(W * W)
  
  f_new = np.exp(f)/np.sum(np.exp(f), axis = 1).reshape(num_train,1)
  f_new[range(num_train),y] -= 1
  dW_new = X.T.dot(f_new)
  dW = dW_new/num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

