import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
from theano.tensor.signal import downsample
import time

# define pool layer for CNN (only 2d max-pool)

class PoolLayer(object) :
	def __init__(self, input, input_shape, pool_shape) :
		# input : theano symbolic variable of input, 4D tensor
		# input_shape : shape of input / (minibatch size, input channel num, image height, image width)
		# pool_shape : size of pool / normally use 2x2

		# max-pool calculation & theano function
		self.output = downsample.max_pool_2d(input, pool_shape)

		# no additional parameter used in pool layer