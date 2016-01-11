import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
from theano.tensor.signal import downsample
import time

# define dropout layer after conv/pool layer

class Dropout(object) :
	def __init__(self, input, input_shape, p) :
		# input : theano symbolic variable of input, 4D tensor
		# input_shape : shape of input / (minibatch size, input channel num, image height, image width)
		# p : probability of dropout, shared variable
		
		srng = T.shared_randomstreams.RandomStreams(int(time.time()))
		p_val = p.get_value()
		select_array = T.cast(srng.binomial(n=1, p=1-p_val, size=input_shape), theano.config.floatX)
		self.output = select_array * input
		