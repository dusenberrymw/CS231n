import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0] # C
  num_examples = X.shape[1] # N
  loss = 0.0
  for i in xrange(num_examples):
    scores = W.dot(X[:, i]) # C length vector
    correct_class_score = scores[y[i]] # scalar
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # delta = 1
      if margin > 0: # mx = max(0, margin)
        # accumulate loss since scores are too close
        loss += margin

        # compute gradient using chain rule
        dmargin = 1 #if margin > 0 else 0 # deriv of max(0,a) function
        dscoresj = 1.0 * dmargin
        dcorrect_class_score = -1.0 * dmargin
        dW[j] += X[:,i] * dscoresj
        dW[y[i]] += X[:,i] * dcorrect_class_score

  # Right now the loss and gradient are sums over all training examples, but
  # we want them to be averages instead so divide each by num_examples.
  loss /= num_examples
  dW /= num_examples

  # Add regularization to the loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it in dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # Note: Completed inline -- see above

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # C x D array; initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[0] # C
  num_examples = X.shape[1] # N

  # Calculate class scores for all examples
  scores = W.dot(X) # C x N array of scores, with one column per example
  # Select the correct score for each column using integer array indexing:
  correct_class_mask = [y, np.arange(num_examples)] 
  correct_class_scores = scores[correct_class_mask] # N length vector
  # We only want to compute margins for the incorrect classes, so compute for
  # all classes and then set margins for correct classes to 0
  margins = scores - correct_class_scores + 1 # C x N array; delta = 1
  margins[correct_class_mask] = 0
  thresh_margins = np.maximum(0, margins)
  loss = np.sum(thresh_margins)
  # Average & add regularization to the loss.
  loss /= num_examples
  loss += 0.5 * reg * np.sum(W * W)
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
  # For each example, the correct_class_score contributes to the final loss
  # as many times as the margin > 0 for that example.
  # So, sum up the number of times the margin is > 0 for each example, and
  # add to the dscores for the correct classifier position in each example
  dthresh_margins = 1.0 # scalar
  dmargins = (margins > 0) * dthresh_margins # C x N array
  dscores = 1.0 * dmargins # C x N array
  dcorrect_class_scores = -1.0 * dmargins.sum(axis=0) # N length vector
  dscores[correct_class_mask] = dcorrect_class_scores
  dW = dscores.dot(X.T) # chain rule with correct dimensions; C x D array
  # Average & add regularization to the gradient
  dW /= num_examples
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

