import numpy as np
import theano
import theano.tensor as T
import math
import time

## define MLP Layer
# input -> hidden layer -> output layer, sigmoid as an hidden activation function
class MLP(object) :
	def __init__(self, input, input_shape, hidden_num, output_num, p) :
		# input : theano symbolic variable of input, 2D tensor 
		# input_shape : shape of input / (mini-batch size, vector length)
		# hidden_num : number of hidden layer nodes
		# output_num : number of output layer nodes, in MNIST case : 10
		# p : dropconnect rate (given by theano shared variable)

		input_num = input_shape[1]

		# initialize W1, W2 (input->hidden, hidden->output) randomly
		rng = np.random.RandomState(int(time.time()))
		w1_bound = math.sqrt(input_num)
		w2_bound = math.sqrt(hidden_num)

		self.W1 = theano.shared(np.asarray(rng.uniform(low=-1.0/w1_bound, high=1.0/w1_bound, size=(input_num, hidden_num)), dtype=theano.config.floatX), name='W11', borrow=True)
		self.W2 = theano.shared(np.asarray(rng.uniform(low=-1.0/w2_bound, high=1.0/w2_bound, size=(hidden_num, output_num)), dtype=theano.config.floatX), name='W2', borrow=True)

		# initialize b1, b2 (input->hidden, hidden->output) randomly
		self.b1 = theano.shared(np.asarray(np.zeros(hidden_num,), dtype=theano.config.floatX), name='b1', borrow=True)
		self.b2 = theano.shared(np.asarray(np.zeros(output_num,), dtype=theano.config.floatX), name='b2', borrow=True)

		# DropConnect
		srng = T.shared_randomstreams.RandomStreams(int(time.time()))
		p_val = p.get_value()
		select_array1 = T.cast(srng.binomial(n=1, p=1-p_val, size=(input_num, hidden_num)), theano.config.floatX)
		select_array2 = T.cast(srng.binomial(n=1, p=1-p_val, size=(hidden_num, output_num)), theano.config.floatX)
		select_vec1 = T.cast(srng.binomial(n=1, p=1-p_val, size=(hidden_num,)), theano.config.floatX)
		select_vec2 = T.cast(srng.binomial(n=1, p=1-p_val, size=(output_num,)), theano.config.floatX)

		# possible hidden calculation
		hid = T.dot(input, self.W1*select_array1)+self.b1*select_vec1

		# leaky relu, softmax
		self.hidden = T.switch(hid<0, 0.01*hid, hid)
		x = T.dot(self.hidden, self.W2*select_array2)+self.b2*select_vec2
		x_prime = x - x.max(axis=1, keepdims=True)
		x_prime2 = x_prime - T.log(T.sum(T.exp(x_prime),axis=1,keepdims=True))
		self.output = T.exp(x_prime2)

		# save parameter of this layer for back-prop convinience
		self.params = [self.W2, self.b2, self.W1, self.b1]
		self.paramins = [hidden_num, hidden_num, input_num, input_num]
		self.input = input