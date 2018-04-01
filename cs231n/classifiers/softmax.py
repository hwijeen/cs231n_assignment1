import numpy as np
from random import shuffle
from past.builtins import xrange

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
  
  N = X.shape[0]
  C = W.shape[1]

  for i in range(N):
    X_i = X[i]    # D,    
    Wx_i = np.dot(X_i, W)    # C,   
    Wx_i -= np.max(Wx_i)    # for numericall stability 
    y_i_hat = np.zeros(C)    # C,

    denom = 0
    for j in range(C):
      y_i_hat[j] = np.exp(Wx_i[j])
      denom += y_i_hat[j]
    y_i_hat /= denom

    for j in range(C):
      dW[:,j] += (1/denom)*np.exp(Wx_i[j]) * X[i]
      if j==y[i]:
        dW[:,j] -= X[i]

    # CELoss
    loss_i = -np.log(y_i_hat[y[i]])
    loss += loss_i
  loss /= N
  dW /= N

  # regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
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
  
  N = X.shape[0]
  C = W.shape[1]

  score = np.dot(X,W)    # NxC
  score -= np.max(score)
  expscore = np.exp(score)    # NxC
  denom = np.sum(expscore, axis=1)    #Nx1 
  numer = expscore[range(N), y]    # 
  
  loss = np.sum(-1*np.log(numer/denom)) / N
  loss += 0.5 * reg * np.sum(W*W)

  correct_class = np.zeros((N, C))
  correct_class[xrange(N),y] = -1
  dW_term1 = X.T.dot(correct_class)
  dW_term2 = (X.T / denom).dot(expscore)
  dW = dW_term1 + dW_term2

  dW /= float(N)
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

