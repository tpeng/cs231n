import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = np.dot(x.reshape(x.shape[0], -1), w) + b
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
  dx = np.dot(dout, w.T).reshape(*x.shape)
  dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
  db = np.sum(dout, axis=0)
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
  out = x.copy()
  out[out < 0] = 0
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
  dx = dout * (x > 0)
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    mu = np.sum(x, axis=0) / N
    xmu = x - mu
    sq = xmu ** 2
    var = np.sum(sq, axis=0) / N
    sqrtvar = np.sqrt(var)
    ivar = 1 / sqrtvar
    xhat = xmu * ivar
    gammax = xhat * gamma
    out = gammax + beta
    running_mean = mu
    running_var = var
    cache = (gammax, xhat, ivar, sqrtvar, var, sq, xmu, mu, gamma, beta)
  elif mode == 'test':
    x = (x - running_mean) / np.sqrt(running_var)
    out  = gamma * x + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  print 'running_mean', running_mean
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  gammax, xhat, ivar, sqrtvar, var, sq, xmu, mu, gamma, beta = cache
  N, D = dout.shape

  dbeta = np.sum(dout, axis=0)
  dgammax = dout
 
  dgamma = np.sum(dgammax * xhat, axis=0)
  dxhat = dgammax * gamma

  dxmu = dxhat * ivar
  # ivar is (D, )
  divar = np.sum(dxhat * xmu, axis=0)

  dsqrtvar = divar * (-1) / (sqrtvar ** 2)

  dvar = dsqrtvar * 0.5 / (np.sqrt(var))

  dsq = dvar * np.ones((N, D)) / N

  dxmu2 = dsq * 2 * xmu

  dx1 = dxmu2 + dxmu

  dmu = - np.sum(dxmu + dxmu2, axis=0)

  dx2 = dmu * np.ones((N, D)) / N

  dx = dx1 + dx2
  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  if mode == 'train':
    mask = np.random.rand(*x.shape) < dropout_param['p'] 
    out = x * mask
    # mask = np.random.choice(x.shape[1], x.shape[1] * dropout_param['p'])
    # out = x[:, mask]
  elif mode == 'test':
    out = x
    mask = None
  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  if mode == 'train':
    # print 'dout.shape', dout.shape
    # print 'mask.shape', mask.shape
    dx = dout * mask
  elif mode == 'test':
    dx = dout.copy()
  return dx


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
  N, _, H, W = x.shape
  F, _, HH, WW = w.shape

  x_pad = ((0, 0), (0, 0), (1, 1), (1, 1))
  x2 = np.pad(x, x_pad, 'constant', constant_values=0)

  stride = conv_param['stride']
  H_ = 1 + (H + 2*conv_param['pad'] - HH) / stride
  W_ = 1 + (W + 2*conv_param['pad'] - WW) / stride

  out = np.zeros((N, F, H_, W_))
  for k in range(N):
    for f in range(0, F):
        for j in range(0, H_):
          for i in range(0, W_):
            out[k, f, j, i] = np.dot(x2[k, :, j*stride:j*stride+HH, i*stride:i*stride+WW].reshape(1, -1), w[f, :, :, :].reshape(1, -1).T) + b[f]

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
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape

  dx, dw, db = None, None, None
  x_pad = ((0, 0), (0, 0), (1, 1), (1, 1))
  x2 = np.pad(x, x_pad, 'constant', constant_values=0)

  stride = conv_param['stride']
  H_ = 1 + (H + 2*conv_param['pad'] - HH) / stride
  W_ = 1 + (W + 2*conv_param['pad'] - WW) / stride

  db = np.sum(np.transpose(dout, (1, 0, 2, 3)).reshape(w.shape[0], -1), axis=1)
  dw = np.zeros((F, C, HH, WW))
  dx = np.zeros((N, C, H, W))
  dx2 = np.zeros_like(x2)

  for f in range(0, F):
    for j in range(0, H_):
      for i in range(0, W_):
        for k in range(0, N):
          region = x2[k, :, j*stride:j*stride+HH, i*stride:i*stride+WW]
          dw[f, :, :, :] += dout[k, f, j, i] * region

  for k in range(0, N):
    for j in range(0, H_):
      for i in range(0, W_):
        v = np.dot(w.reshape(F, -1).T, dout[k, :, j, i]).reshape((C, HH, WW))
        dx2[k, :, j*stride:j*stride+HH, i*stride:i*stride+WW] += v

  dx = dx2[:, :, 1:-1, 1:-1]
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
  N, C, H, W = x.shape
  stride = pool_param['stride']
  HH = H / stride
  WW = W / stride

  out = np.zeros((N, C, HH, WW))
  for j in range(HH):
    for i in range(WW):
      out[:, :, j, i] = np.max(x[:, :, j*stride: j*stride + stride, i *stride: i*stride+stride], axis=(-1, -2))
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
  x, pool_param = cache
  dx = np.zeros_like(x)
  stride = pool_param['stride']
  N, C, H, W = x.shape
  HH = H / stride
  WW = W / stride

  out = np.zeros((N, C, HH, WW))
  dx = np.zeros_like(x)

  for k in range(N):
    for f in range(C):
      for j in range(HH):
        for i in range(WW):
          v = x[k, f, j*stride: j*stride + stride, i *stride: i*stride+stride]
          _j, _i = np.unravel_index(np.argmax(v), v.shape)
          dx[k, f, j *stride + _j, i*stride + _i] = dout[k, f, j, i]

  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape
  out = np.zeros_like(x)
  cache = []

  for i in range(C):
    _x = x[:, i, :, :].reshape(N, -1) 
    _out, _cache = batchnorm_forward(_x, gamma[i], beta[i], bn_param)
    out[:, i, :, :] = _out.reshape((N, H, W))
    cache.append(_cache)
  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  N, C, H, W = dout.shape
  dgamma = np.zeros(C)
  dbeta = np.zeros(C)

  dx = np.zeros_like(dout)

  for i in range(C):
    _dx, _dgamma, _dbeta = batchnorm_backward(dout[:, i, :, :].reshape(N, -1), cache[i])
    dx[:, i, :, :] = _dx.reshape((N, H, W))
    dgamma[i] = np.sum(_dgamma, axis=0)
    dbeta[i] = np.sum(_dbeta, axis=0)
  return dx, dgamma, dbeta
  

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
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
