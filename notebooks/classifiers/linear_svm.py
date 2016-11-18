import numpy as np
from random import shuffle

def svm_loss_bias_forloop(W, b, X, y, reg, delta = 1):
  """
  Multiclass Linear SVM. Naive implementation.

  Input feature have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - b: A numpy array of shape (C,) containing bais.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - delta: expectd functional margin from the decision plane
  Returns a tuple of:
  - loss as single float
  - d_W as the gradient with respect to weights W; an array of same shape as W
  - d_b as the gradient with respect to weights b; an array of same shape as b 
  """    
  # initialize the returned results
  loss = 0.0
  d_W = np.zeros(W.shape) 
  d_b = np.zeros(b.shape)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  for i in xrange(num_train):
    # compute the classification scores for a single image
    scores = X[i].dot(W) + b
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        # compute the loss for this image
        if j == y[i]:
            continue
        margin = scores[j] - correct_class_score + delta 
        if margin > 0:
            loss += margin
            # compute the gradient for this image
            d_W[:, j] += X[i, :].T
            d_b[j] += 1
            d_W[:, y[i]] -= X[i, :].T
            d_b[y[i]] -= 1
            
  # Right now the loss is a sum over all training examples
  # We need it to be an average instead so we divide by num_train.
  loss /= num_train  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  # Do the same for d_W and d_b
  d_W /= num_train
  d_W += reg*W

  d_b /= num_train
    
  return loss, d_W, d_b  


def svm_loss_forloop(W, X, y, reg, delta=1):
  """
  Multiclass Linear SVM. For loop implmentation with bias trick.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - delta: expectd functional margin from the decision plane
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  ################################################################################
  # You implementation                                                           #
  # Use the ahove svm_loss_bias_forloop implementation as reference              #
  ################################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg, delta=1):
  """
  Multi-class linear SVM. Vectorized implmentation.

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - margin: Expected functional margin from the decision plane
  - delta: expectd functional margin from the decision plane

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # Understand this implementation                                            #
  #############################################################################
  # Hint: check how numpy broadcasting and advanced indexing are used
  # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
  # This allows selection of arbitrary items in the array based on their N-dimensional index. Each integer array represents a number of indexes into that dimension.

  # Get dims
  D = X.shape[1]
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)

  correct_scores = scores[np.arange(num_train), y].reshape(-1, 1) # using the fact that all elements in y are < C == num_classes
  mat = scores - correct_scores + delta 
  mat[np.arange(num_train), y] = 0 # accounting for the j=y_i term we shouldn't count (subtracting 1 makes up for it since w_j = w_{y_j} in this case)
  
  # Compute max
  thresh = np.maximum(np.zeros((num_train, num_classes)), mat)
  # Compute loss as double sum
  loss = np.sum(thresh)
  loss /= num_train
    
  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)

  # Binarize into integers
  binary = thresh
  binary[thresh > 0] = 1

  row_sum = np.sum(binary, axis=1)
  binary[range(num_train), y] = -row_sum[range(num_train)]
  dW = np.dot(X.T, binary)

  # Divide
  dW /= num_train

  # Regularize
  dW += reg*W
  
  return loss, dW

class LinearSVM:

  def __init__(self):
    self.W = None
    
  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      self.W = np.random.randn(dim, num_classes) * 0.001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      # Step One: Implement Stochastic
      #########################################################################
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################

      # Step Two: Implement Gradient
      # Simply call self.loss (which calls svm_loss_vectorized) to evaluate loss and gradient

      # Step Three: Implement Descent
      # Simply update the weights using the gradient and the learning rate.          #

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    y_pred = np.zeros(X.shape[0])
    
    ###########################################################################
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################

    return y_pred

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)




