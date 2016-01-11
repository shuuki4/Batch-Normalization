import theano.tensor as T
import theano
import numpy as np
import cPickle
import gzip
import ConvLayer
import PoolLayer
import Dropout
import MLP

# fetch data from file
f = gzip.open("mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_num = train_x.shape[0]
valid_num = valid_x.shape[0]
test_num = test_x.shape[0]

# initially normalize all datas in mean 0, std 1
for i in range(train_num) :
	train_x[i,:] = np.add(train_x[i,:], -np.mean(train_x[i,:]))
	train_x[i,:] /= np.std(train_x[i,:])
for i in range(valid_num) :
	valid_x[i,:] = np.add(valid_x[i,:], -np.mean(valid_x[i,:]))
	valid_x[i,:] /= np.std(valid_x[i,:])
for i in range(test_num) :
	test_x[i,:] = np.add(test_x[i,:], -np.mean(test_x[i,:]))
	test_x[i,:] /= np.std(test_x[i,:])
# reshape
train_x = np.reshape(train_x, (train_num, 1, 28, 28))
valid_x = np.reshape(valid_x, (valid_num, 1, 28, 28))
test_x = np.reshape(test_x, (test_num, 1, 28, 28))

### simple 3-layer convolutional layer for test : dropout & dropconnect included

# parameters
mini_batch_size = 50
input_shape = (mini_batch_size, 1, 28, 28)
learning_rate = 0.002
dropout_rate = 0.25
dropconnect_rate = 0.5
dropout_var = theano.shared(value=dropout_rate, name="dropout")
dropconnect_var = theano.shared(dropconnect_rate, name="dropconnect")

# layer structure
input = T.tensor4(name='input')

convlayer1 = ConvLayer.ConvLayer(input, input_shape, filter_shape=(32, 1, 5, 5))
poollayer1 = PoolLayer.PoolLayer(convlayer1.output, input_shape=(mini_batch_size, 32, 24, 24), pool_shape=(2,2))
dropout1 = Dropout.Dropout(poollayer1.output, input_shape=(mini_batch_size, 32, 12, 12), p=dropout_var)
convlayer2 = ConvLayer.ConvLayer(dropout1.output, input_shape=(mini_batch_size, 32, 12, 12), filter_shape=(64, 32, 3, 3))
poollayer2 = PoolLayer.PoolLayer(convlayer2.output, input_shape=(mini_batch_size, 64, 10, 10), pool_shape=(2,2))
convlayer3 = ConvLayer.ConvLayer(poollayer2.output, input_shape=(mini_batch_size, 64, 5, 5), filter_shape=(128, 64, 3, 3))
mlp_input = T.reshape(convlayer3.output, (mini_batch_size, 128*3*3), ndim=2)
MLPlayer = MLP.MLP(mlp_input, input_shape=(mini_batch_size, 128*3*3), hidden_num=800, output_num=10, p=dropconnect_var)

y = T.matrix('y') # real one-hot indexes
cost = T.nnet.categorical_crossentropy(MLPlayer.output, y).sum()

params = MLPlayer.params + convlayer3.params + convlayer2.params + convlayer1.params
grad = T.grad(cost, params)
updates= [(param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params, grad)]

f = theano.function([input, y], cost, updates=updates) # for train
test_f = theano.function([input], MLPlayer.output) # for validation, test

# real train area
for epoch in range(50) : # to change epoch number, fix this number
	random_idx = np.random.permutation(train_num)
	for mb_idx in range(train_num/mini_batch_size) :
		# make testset
		now_input = np.zeros((mini_batch_size, 1, 28, 28), dtype=theano.config.floatX)
		for i in range(mini_batch_size) : now_input[i,0,:,:] = train_x[random_idx[mb_idx*mini_batch_size+i],:,:]
		# make y set
		now_y = np.zeros((mini_batch_size, 10), dtype=theano.config.floatX)
		for i in range(mini_batch_size) : now_y[i, train_y[random_idx[mb_idx*mini_batch_size+i]]] = 1.0	
		# run
		nowcost = f(now_input, now_y)

	# validation data check
	miss_count = 0.0
	dropout_var.set_value(0.0)
	dropconnect_var.set_value(0.0)

	for mb_idx in range(valid_num/mini_batch_size) :
		# make testset
		now_input = np.zeros((mini_batch_size, 1, 28, 28), dtype=theano.config.floatX)
		for i in range(mini_batch_size) : now_input[i,0,:,:] = valid_x[mb_idx*mini_batch_size+i,:,:]
		# run
		result = test_f(now_input).argmax(axis=1)
		for i in range(mini_batch_size) :
			if result[i] != valid_y[mb_idx*mini_batch_size+i] : miss_count+=1.0
	print "Epoch %d : Validation Miss rate %lf" % (epoch+1, miss_count/valid_num)

# final : test set miss rate check

miss_count = 0.0
for mb_idx in range(test_num/mini_batch_size) :
	# make testset
	now_input = np.zeros((mini_batch_size, 1, 28, 28), dtype=theano.config.floatX)
	for i in range(mini_batch_size) : now_input[i,0,:,:] = test_x[mb_idx*mini_batch_size+i,:,:]
	# run
	result = test_f(now_input).argmax(axis=1)
	for i in range(mini_batch_size) :
		if result[i] != test_y[mb_idx*mini_batch_size+i] : miss_count+=1.0
print "Test Data Miss rate : %lf" % (miss_count/test_num)