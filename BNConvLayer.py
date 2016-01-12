import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
import time
import math
import BatchNormalization

# define convolution layer for BNCNN

class BNConvLayer(object) :
	def __init__(self, input_shape, filter_shape, border_mode="valid", BN=True) :

		# input_shape : shape of input / (minibatch size, input channel num, image height, image width)
		# filter_shape : shape of filter / (# of new channels to make, input channel num, filter height, filter width)
		# BN : boolean value that determines to apply Batch Normalization or not

		self.BN = BN
		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.border_mode = border_mode

		# initialize W (weight) randomly
		rng = np.random.RandomState(int(time.time()))
		w_bound = math.sqrt(filter_shape[1] * filter_shape[2] * filter_shape[3])
		self.W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=filter_shape), dtype=theano.config.floatX), name='W', borrow=True)
		# initialize b (bias) with zeros
		self.b = theano.shared(np.asarray(np.zeros(filter_shape[0],), dtype=theano.config.floatX), name='b', borrow=True)

		if BN == True :
			# calculate appropriate input_shape
			new_shape = list(input_shape)
			new_shape[1] = filter_shape[0]
			if border_mode == "valid" :
				new_shape[2] -= (filter_shape[2]-1)
				new_shape[3] -= (filter_shape[3]-1)
			elif border_mode == "full" :
				new_shape[2] += (filter_shape[2]-1)
				new_shape[3] += (filter_shape[3]-1)
			new_shape = tuple(new_shape)
			self.BNlayer = BatchNormalization.BatchNormalization(new_shape, mode=1)

		# save parameter of this layer for back-prop convinience
		if BN == True : self.params = [self.W] + self.BNlayer.params
		else : self.params = [self.W, self.b]

		insize = input_shape[1] * input_shape[2] * input_shape[3]
		self.paramins = [insize, insize]

	def set_runmode(self, run_mode) :
		self.BNlayer.set_runmode(run_mode) 

	def get_result(self, input) :

		if self.BN == True :
			out = conv.conv2d(input, self.W, image_shape=self.input_shape, filter_shape=self.filter_shape, border_mode=self.border_mode)
			out = self.BNlayer.get_result(out)
		else :
			out = conv.conv2d(input, self.W, image_shape=self.input_shape, filter_shape=self.filter_shape, border_mode=self.border_mode) + self.b.dimshuffle('x', 0, 'x', 'x')

		# Leaky ReLU
		self.output = T.switch(out<0, 0.01*out, out)
		return self.output