from builtins import range
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
    dW = np.zeros(W.shape) #init the derivatives to 0

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #p = np.zeros((num_train, num_classes))
    
    for i in range(num_train):
      f = X[i] @ W
      f -= np.max(f)
      p = np.exp(f) / np.sum(np.exp(f))
      loss -= np.log(p[y[i]])

       ## computing dW:
      for j in range (num_classes):
          dW[:,j] += (p[j] - (y[i] == j))* X[i]
         
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += (reg * 2 * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    f_exp_sums = 0 

    f = X @ W
    ## trick with logC:
    f -= np.expand_dims(np.max(f, axis=1), axis=1)
    ## compute the denominator of zj:
    f_exp_sums = np.expand_dims(np.sum(np.exp(f), axis=1), axis=1)
    ## compute the probabilities
    p = np.exp(f) / f_exp_sums
    
    ## dW computation: (p[j] - (y[i] == j))* X[i]
    p_mask = np.zeros(p.shape)
    p_mask[np.arange(num_train), y] = 1
    dW = (X.T @ (p - p_mask) / num_train) + (reg * 2 * W)
    
    loss_matrix = p[range(p.shape[0]),y]
    loss_matrix = -np.log(loss_matrix)
    loss = (np.sum(loss_matrix) / num_train) + (reg * np.sum(W * W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
