import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k) # so NOT unrolled
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = np.dot(x.reshape((x.shape[0], -1)), w) + b # reshape x to N x D
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape((x.shape[0], -1)).T, dout)
  db = np.sum(dout, axis=0).reshape(b.shape) # account for (1,M) or (M,) shapes
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # out = max(0, x)
  # dout/dx = 1 if x > 0, else 0
  # dupstream/dx = (1 if x > 0, else 0) * dout
#  dx = (x >= 0) * dout  # likely slower due to creation of extra array
  dx = dout
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  padded_x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant') 

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  Hout = 1 + (H + 2 * pad - HH) / stride
  Wout = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N,F,Hout,Wout))

#  print 'H: %d, W: %d, HH: %d, WW: %d' % (H, W, HH, WW)
#  print 'Hout: %d, Wout: %d' % (Hout, Wout)

  # naive approach
#  for n in xrange(N):
#    for f in xrange(F):
#      for hout in xrange(Hout):
#        h0 = hout * stride
#        for wout in xrange(Wout):
#          w0 = wout * stride
#          out[n,f,hout,wout] = (np.sum(padded_x[n,:,h0:h0+HH,w0:w0+WW] * w[f])
#                                + b[f])
        
  # partially vectorized
#  for n in xrange(N):
#    for hout in xrange(Hout):
#      h0 = hout * stride
#      #print 'h0: %d, h1: %d' % (h0, h1)
#      for wout in xrange(Wout):
#        w0 = wout * stride
#        #print 'w0: %d, w1: %d' % (w0, w1)
#        #out[n,:,hout,wout] = (np.dot(w.reshape(w.shape[0], -1), 
#        #                         padded_x[n,:,h0:h0+HH,w0:w0+WW].reshape(-1))
#        #                      + b)
#        ax = tuple(range(1,len(w.shape))) # all but first axis
#        out[n,:,hout,wout] = (np.sum(w * padded_x[n,:,h0:h0+HH,w0:w0+WW], 
#                                     axis=ax) + b)

  # partially vectorized with unrolled matrices
  rwT = w.reshape(w.shape[0], -1).T # (C*HH*WW, F)
  for hout in xrange(Hout):
    h0 = hout * stride
    for wout in xrange(Wout):
      w0 = wout * stride
      rx = padded_x[:,:,h0:h0+HH,w0:w0+WW].reshape(N, -1) # (N, C*HH*WW)
      out[:,:,hout,wout] = np.dot(rx, rwT) + b # (N, F, 1, 1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  padded_x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant') 

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  Hout = 1 + (H + 2 * pad - HH) / stride
  Wout = 1 + (W + 2 * pad - WW) / stride

  dx = np.zeros_like(x)
  dpadded_x = np.zeros_like(padded_x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

#  # naive approach
#  for n in xrange(N):
#    for f in xrange(F):
#      for hout in xrange(Hout):
#        h0 = hout * stride
#        for wout in xrange(Wout):
#          w0 = wout * stride
#          #out[n,f,hout,wout] = (np.sum(padded_x[n,:,h0:h0+HH,w0:w0+WW] * w[f])
#          #                      + b[f])
#          dpadded_x[n,:,h0:h0+HH,w0:w0+WW] += w[f] * dout[n,f,hout,wout]
#          dw[f] += padded_x[n,:,h0:h0+HH,w0:w0+WW] * dout[n,f,hout,wout]
#          db[f] += dout[n,f,hout,wout]
#
#  # unpad
#  dx = dpadded_x[:, :, pad:-pad, pad:-pad]

  # partially vectorized with unrolled matrices
  rw = w.reshape(w.shape[0], -1)
  for hout in xrange(Hout):
    h0 = hout * stride
    for wout in xrange(Wout):
      w0 = wout * stride
      rx = padded_x[:,:,h0:h0+HH,w0:w0+WW].reshape(N, -1)
      dfeat_map = dout[:,:,hout,wout]
      dpadded_x[:,:,h0:h0+HH,w0:w0+WW] += np.dot(dfeat_map, rw).reshape(N, C, HH, WW)
      dw += np.dot(dfeat_map.T, rx).reshape(dw.shape)
      db += dfeat_map.sum(axis=0)

  # unpad
  dx = dpadded_x[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']

  N, C, H, W = x.shape
  Hout = (H - pool_height) / stride + 1
  Wout = (W - pool_width) / stride + 1
  out = np.zeros((N,C,Hout,Wout))

#  # naive approach
#  for n in xrange(N):
#    for c in xrange(C):
#      for hout in xrange(Hout):
#        h0 = hout * stride
#        for wout in xrange(Wout):
#          w0 = wout * stride
#          out[n,c,hout,wout] = np.max(x[n,c,h0:h0+pool_height,w0:w0+pool_width])

  # partially vectorized
  for hout in xrange(Hout):
    h0 = hout * stride
    for wout in xrange(Wout):
      w0 = wout * stride
      patch = x[:,:,h0:h0+pool_height,w0:w0+pool_width]
      out[:,:,hout,wout] = np.max(patch, axis=(2,3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']

  N, C, H, W = x.shape
  Hout = (H - pool_height) / stride + 1
  Wout = (W - pool_width) / stride + 1

  dx = np.zeros_like(x)

  # naive approach
  for n in xrange(N):
    for c in xrange(C):
      for hout in xrange(Hout):
        h0 = hout * stride
        for wout in xrange(Wout):
          w0 = wout * stride
          ind = np.argmax(x[n,c,h0:h0+pool_height,w0:w0+pool_width])
          ix = np.unravel_index(ind, (pool_height, pool_width))
          dx[n,c,h0:h0+pool_height,w0:w0+pool_width][ix] = dout[n,c,hout,wout]
          
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True)) # subtract max for stability
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  # add smoothing to avoid log(0)
  loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-20)) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

