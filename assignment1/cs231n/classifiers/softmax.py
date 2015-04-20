import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  dim, num_examples = X.shape
  classes = W.shape[0]
  for i in xrange(num_examples):
    correct_class = y[i]
    fi = W.dot(X[:,i]) # C length array of unnormalized log class probabilities
    fi -= np.max(fi) # subtract max for numerical stability with e^x operation
    prob_fyi = np.exp(fi[correct_class]) # scalar
    probs = np.exp(fi) # C length vector
    norm_prob_fyi = prob_fyi / np.sum(probs) # scalar
    loss += -np.log(norm_prob_fyi) # scalar

    dnorm_prob_fyi = -1/norm_prob_fyi # scalar
    dprob_fyi = 1/np.sum(probs) * dnorm_prob_fyi # scalar
    dprobs = -prob_fyi/(np.sum(probs)**2) * dnorm_prob_fyi # scalar
    dfi = np.exp(fi) * dprobs # C length array
    dfi[correct_class] += np.exp(fi[correct_class]) * dprob_fyi
    dW += np.outer(dfi, X[:,i]) # C x D array

  # average and add regularization
  loss /= num_examples
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_examples
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  num_examples = X.shape[1]
  correct_class_mask = [y, np.arange(num_examples)] 

  f = W.dot(X) # C x N array of unnormalized log class probabilites
  f -= np.max(f, axis=0) # subtract max probs for numerical stability with e^x
  prob_fy = np.exp(f[correct_class_mask]) # N length vec of unnormalized probs
  probs = np.exp(f) # C x N array of unnormalized probabilites
  norm_prob_fy = prob_fy / np.sum(probs, axis=0) # N length vector
  norm_prob_fy[np.where(norm_prob_fy == 0)] = 1e-10 # for numerical stability
  loss = np.sum(-np.log(norm_prob_fy))
  # Average & add regularization
  loss /= num_examples
  loss += 0.5 * reg * np.sum(W*W)

  # "Automatic Differentiation" (AD) -> systematically applying the chain rule
  # for each elementary operation using simple, known derivatives in order to 
  # compute full, complex derivatives, rather than  manually deriving the 
  # full derivatives.
  # -Easier & faster to implement.
  # This is just a simple example of 'reverse mode' AD.
  # http://arxiv.org/abs/1502.05767
  dnorm_prob_fy = -1.0/norm_prob_fy # N length vector
  dprob_fy = 1/np.sum(probs, axis=0) * dnorm_prob_fy # N length vector
  dprobs = -prob_fy/(np.sum(probs, axis=0)**2) * dnorm_prob_fy # N length vec
  df = probs * dprobs # C x N array
  df[correct_class_mask] += prob_fy * dprob_fy
  dW = df.dot(X.T) # C x D array; dloss/dW = X.T * df
  # Average & add regularization
  dW /= num_examples
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

