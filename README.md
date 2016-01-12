# Batch-Normalization

## Description
This repo contains an implementation of [Batch Normalization](http://arxiv.org/abs/1502.03167) Layer by Theano. Layer Performance is tested by MNIST Dataset, by simple 3 conv-layer CNN.
- BatchNormalization.py : Batch Normalization Layer. Supports both normal/CNN mode. Should set set_runmode(1) before test, and set_runmode(0) before train.
- BNConvLayer.py : Convolution Layer with BN layer before activation. Activation : Leaky ReLU
- BNMLP.py : 3-Layer MLP with BN layer before hidden layer. Activation : Leaky ReLU
- BNaddedCNN.py : MNIST Performance checker. Uses 3-conv layer (channel : 32->64->128) CNN with Batch Normalization.
- normalCNN.py : MNIST Performace checker control group. Uses same network with BNaddedCNN - excluding Batch Normalization and including Dropout / Dropconnect.
- MLP.py, ConvLayer.py, Dropout.py, MLP.py, PoolLayer.py : Layers needed to make CNN structure


## Further Explanation
Further explanation of this code and the theory of batch normalization concept can be found on [my blog](http://shuuki4.wordpress.com). It is written in Korean. 
