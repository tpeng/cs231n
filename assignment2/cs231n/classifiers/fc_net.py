import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from cs231n.layer_utils import affine_batchnorm_relu_forward
from cs231n.layer_utils import affine_batchnorm_relu_backward
from cs231n.layer_utils import affine_batchnorm_forward
from cs231n.layer_utils import affine_batchnorm_backward

def softmax(l):
  if len(l.shape) == 1:
    l -= l.max()
    return np.exp(l) / np.sum(np.exp(l))
  else:
    l -= np.max(l, axis=1)[:, np.newaxis]
    return np.exp(l) / np.sum(np.exp(l), axis=1)[:, np.newaxis]


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    self.params['W1'] = np.random.normal(0, weight_scale, input_dim * hidden_dim).reshape(input_dim, hidden_dim)
    self.params['W2'] = np.random.normal(0, weight_scale, hidden_dim * num_classes).reshape(hidden_dim, num_classes)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2']

    a1_out, a1_cache = affine_relu_forward(X, W1, b1)
    scores = np.dot(a1_out, W2) + b2

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)

    dW2 = np.dot(a1_out.T, dout)
    dW2 += self.reg * W2
    dB2 = np.sum(dout, axis=0)

    dX, dW1, dB1 = affine_relu_backward(np.dot(dout, W2.T), a1_cache)
    dW1 += self.reg * W1

    grads = {}
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = dB1
    grads['b2'] = dB2

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.cache = {}
    self.dropout_cache = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for layer_idx in xrange(0, self.num_layers):
      idx = layer_idx + 1
      if layer_idx == 0:
        fan_in = input_dim
        fan_out = hidden_dims[layer_idx]
      elif layer_idx == len(hidden_dims):
        fan_in = hidden_dims[layer_idx-1]
        fan_out = num_classes
      else:
        fan_in = hidden_dims[layer_idx-1]
        fan_out = hidden_dims[layer_idx]
      self.params['W%s' % idx] = np.random.normal(0, weight_scale, fan_in * fan_out).reshape(fan_in, fan_out)
      self.params['b%s' % idx] = np.zeros(fan_out)
      if self.use_batchnorm:
        self.params['gamma%s' % idx] = np.ones(fan_out)
        self.params['beta%s' % idx] = np.zeros(fan_out)

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    _input = X.copy()
    for idx in xrange(0, self.num_layers-1):
      layer_idx = idx + 1
      if self.use_batchnorm:
        out, cache = affine_batchnorm_relu_forward(
          _input, 
          self.params['W%d' % layer_idx], 
          self.params['b%d' % layer_idx],
          self.params['gamma%d' % layer_idx],
          self.params['beta%d' % layer_idx],
          self.bn_params[idx]
          )
      else:
        out, cache = affine_relu_forward(
          _input, 
          self.params['W%d' % layer_idx], 
          self.params['b%d' % layer_idx])
      self.cache[layer_idx] = cache
      if self.use_dropout:
        _input, do_cache = dropout_forward(out, self.dropout_param)
        self.dropout_cache[layer_idx] = do_cache
      else:
        _input = out.copy()

    if self.use_batchnorm:
      scores, cache = affine_batchnorm_forward(_input,
        self.params['W%d' % self.num_layers],
        self.params['b%d' % self.num_layers],
        self.params['gamma%d' % self.num_layers],
        self.params['beta%d' % self.num_layers],
        self.bn_params[-1])
    else:
      scores, cache = affine_forward(_input,
        self.params['W%d' % self.num_layers],
        self.params['b%d' % self.num_layers])

    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, dout = softmax_loss(scores, y)
    grads = {}

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    if self.use_batchnorm:
      dout, dw, db, dgamma, dbeta = affine_batchnorm_backward(dout, cache)
      grads['W%s' % self.num_layers] = dw
      grads['b%s' % self.num_layers] = db
      grads['gamma%s' % self.num_layers] = dgamma
      grads['beta%s' % self.num_layers] = dbeta
    else:
      grads['W%s' % self.num_layers] = np.dot(_input.T, dout)
      grads['b%s' % self.num_layers] = np.sum(dout, axis=0)
      # get the dout right before the last affine layer
      dout = np.dot(dout, self.params['W%d' % self.num_layers].T)
    for idx in range(self.num_layers-1, 0, -1):
      if self.use_dropout:
        dout = dropout_backward(dout, self.dropout_cache[idx])
      next_layer_idx = idx + 1
      if self.use_batchnorm:
        dout, _dw, _db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, self.cache[idx])
        grads['gamma%d' % idx] = dgamma
        grads['beta%d' % idx] = dbeta
      else:
        dout, _dw, _db = affine_relu_backward(dout, self.cache[idx])
      grads['W%d' % idx] = _dw
      grads['b%d' % idx] = _db

    for idx in range(self.num_layers, 0, -1):
      w = self.params['W%s' % idx]
      loss += 0.5 * self.reg * np.sum(w * w)
      grads['W%s' % idx] += self.reg * w

    return loss, grads
