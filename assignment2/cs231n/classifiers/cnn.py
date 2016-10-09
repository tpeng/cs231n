import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    # self.params['W1'] = np.random.normal(0, weight_scale, num_filters * input_dim[0] * filter_size * filter_size).reshape((num_filters, \
    #     input_dim[0], filter_size, filter_size))
    # self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size ,filter_size) * np.sqrt(2.0 / (filter_size*filter_size*input_dim[0]))
    # print np.sqrt(2.0/(filter_size*filter_size*input_dim[0]))
    # self.params['W1'] = np.random.normal(0, np.sqrt(2.0/(filter_size*filter_size*input_dim[0])), num_filters * input_dim[0] * filter_size * filter_size).reshape((num_filters, \
    #     input_dim[0], filter_size, filter_size))
    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    # self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size ,filter_size) * np.sqrt(1.0 / (filter_size * filter_size * input_dim[0]))
    self.params['b1'] = np.zeros(num_filters)

    # 2x2 max pool
    dim = num_filters * input_dim[1] / 2 * input_dim[2] / 2
    # self.params['W2'] = np.random.normal(0, np.sqrt(2.0/dim), dim * hidden_dim).reshape(dim, hidden_dim)
    # self.params['W2'] = np.random.normal(0, weight_scale, dim * hidden_dim).reshape(dim, hidden_dim)
    # self.params['W2'] = np.random.randn(dim, hidden_dim) * np.sqrt(2.0 / dim)
    self.params['W2'] = np.random.randn(dim, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)

    # self.params['W3'] = np.random.normal(0, weight_scale, hidden_dim * num_classes).reshape(hidden_dim, num_classes)
    # self.params['W3'] = np.random.randn(hidden_dim, num_classes) / np.sqrt(hidden_dim)
    # self.params['W3'] = np.random.randn(hidden_dim, num_classes) * np.sqrt(1.0 / hidden_dim)
    # self.params['W3'] = np.random.normal(0, np.sqrt(2.0/hidden_dim), hidden_dim * num_classes).reshape(hidden_dim, num_classes)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    a1, a1_cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    a2, a2_cache = affine_relu_forward(a1, self.params['W2'], self.params['b2'])
    scores, cache = affine_forward(a2, self.params['W3'], self.params['b3'])

    if y is None:
      return scores

    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)
    loss += 0.5 * self.reg * np.sum(W3 * W3)

    grads = {}
    dout, grads['W3'], grads['b3'] = affine_backward(dout, cache)
    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, a2_cache)
    dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, a1_cache)

    grads['W3'] += self.reg * W3
    grads['W2'] += self.reg * W2
    grads['W1'] += self.reg * W1

    return loss, grads
