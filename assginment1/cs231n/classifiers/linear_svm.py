import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]  # += because all training example!!
        dW[:, y[i]] -= X[i]  # CLEVER!!
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)    
  dW += reg * W    # no coefficient because of 0.5 above

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # num_train = X.shape[0]
  # num_classes = W.shape[1]

  # Wy = np.zeros((num_train, num_train))
  # for i, y_idx in enumerate(y):
  #   Wy[:,i] = W[:,y_idx]

  # loss = np.sum(np.dot(X,W), axis=1) - num_classes * np.sum(np.dot(X, Wy), axis=1) + num_train*(num_classes-1)
  # loss = np.sum(loss[loss>0]) / num_train
  # loss += 0.5 * reg * np.sum(W * W)

  num_train = X.shape[0]    
  num_classes = W.shape[1]    # W는  input과 transformed의 dimension!! not # of train

  scores = np.dot(X, W)    # NxC  
  correct_class_score = scores[np.arange(num_train), y]    # indexes coulb be given as two separate arrays!    # Nx1
  temp = np.maximum(scores - correct_class_score.reshape(num_train,1) + 1.0, 0)    # NxC    # vector into array for broadcasting
  #temp[np.arange(num_train), y] = 0    # -num_train과 equivalent
  loss = (np.sum(temp) - num_train) / num_train
  loss += 0.5 * reg * np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  dscores = np.zeros_like(scores)
  dscores[temp>0] = 1
  dscores[np.arange(num_train), y] -= np.sum(dscores,axis=1)

  dW = np.dot(X.T, dscores)
  dW /= num_train 
  dW += reg * W

  pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
