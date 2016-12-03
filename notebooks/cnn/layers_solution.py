import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # Affine forward pass, reshaping the input into rows.                       #
  #############################################################################
  N = x.shape[0]
  out = x.reshape(N, np.prod(x.shape[1:])).dot(w)+b

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
    - b: Biases, of shape (M,)


  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # Affine backward pass.                                                     #
  #############################################################################
  N = x.shape[0]

  # Your code ************************************
  # compute dx from dout and w 
  dx = dout.dot(w.T).reshape(x.shape)
  # compute dw from x and dout
  dw = x.reshape(N, np.prod(x.shape[1:])).T.dot(dout)
  # cmopute db from dout
  db = np.sum(dout, axis=0)
  # End of your code ***************************** 

  return dx, dw, db

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

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

  #############################################################################
  # Convolutional forward pass                                                #
  #############################################################################
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  out = np.zeros((N, F, Hp, Wp))

  # Add padding around each 2D image
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for i in xrange(N): # ith example
    for j in xrange(F): # jth filter

      # Convolve this filter over windows
      for k in xrange(Hp):
        hs = k * stride
        for l in xrange(Wp):
          ws = l * stride
        
          # get the local window from padded to do correleation with w[j]
          window = padded[i, :, hs:hs+HH, ws:ws+WW]
        
          # Compute out[i, j, k, l] with convolution (in fact, correlation)
          out[i, j, k, l] = np.sum(window*w[j])
        
          # Add bais to out[i, j, k, l]
          out[i, j, k, l] += b[j]

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
  #############################################################################
  # Convolutional backward pass                                               #
  #############################################################################
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # "padded" is a padded map for x
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  # "padded_dx" is a padded gradient map for x
  padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for i in xrange(N): # ith example
    for j in xrange(F): # jth filter
      # Convolve this filter over windows
      for k in xrange(Hp):
        hs = k * stride
        for l in xrange(Wp):
          ws = l * stride

          # get the local window from padded to do correleation with w[j]
          window = padded[i, :, hs:hs+HH, ws:ws+WW]
        
          # Your code ************************************
          # update padded_dx[i, :, hs:hs+HH, ws:ws+WW] by "spraying" dout[i, j, k, l] onto it (through w[j])
          padded_dx[i, :, hs:hs+HH, ws:ws+WW] += w[j] * dout[i, j, k, l]
          # update dw[j] by "spraying" dout[i, j, k, l] onto it (through window)
          dw[j] += window*dout[i, j, k, l]
          # update db[j]. Since the partial derivative of dout[i, j, k, l] w.r.t db[j] is alwasy 
          db[j] += dout[i, j, k, l]
          # End of your code *****************************

  # "Unpad" the gradient map of x
  dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
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

  #############################################################################
  # Max pooling forward pass                                                  #
  #############################################################################
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = 1 + (H - HH) / stride
  Wp = 1 + (W - WW) / stride

  out = np.zeros((N, C, Hp, Wp))

  for i in xrange(N):
    # Need this; apparently we are required to max separately over each channel
    for j in xrange(C):
      for k in xrange(Hp):
        hs = k * stride
        for l in xrange(Wp):
          ws = l * stride
        
          window = x[i, j, hs:hs+HH, ws:ws+WW]
          out[i, j, k, l] = np.max(window)

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

  #############################################################################
  # Max pooling backward pass                                                 #
  #############################################################################
  x, pool_param = cache
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = 1 + (H - HH) / stride
  Wp = 1 + (W - WW) / stride

  dx = np.zeros_like(x)

  for i in xrange(N):
    for j in xrange(C):
      for k in xrange(Hp):
        hs = k * stride
        for l in xrange(Wp):
          ws = l * stride

          window = x[i, j, hs:hs+HH, ws:ws+WW]
          m = np.max(window)

          # Your code ************************************
          # update dx[i, j, hs:hs+HH, ws:ws+WW] by "spraying" dout[i, j, k, l] onto it (through a binary mask window == m)
          dx[i, j, hs:hs+HH, ws:ws+WW] += (window == m) * dout[i, j, k, l]
          # End of your code *****************************

  return dx

def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  #############################################################################
  # ReLU forward pass.                                                        #
  #############################################################################
  out = x*(x>0)

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
  x = cache
  #############################################################################
  # ReLU backward pass.                                                       #
  #############################################################################
  # Your code ************************************
  # pass dout to dx with a binary mask x > 0
  dx = dout * (x>0)
  # End of your code *****************************
    
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
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-17)) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

