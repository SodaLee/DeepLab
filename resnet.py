import tensorflow as tf

def conv_layer(inputs, fsize, name, stride = [1,1,1,1],
	rate = 1, padding = "SAME", use_bn = True, activate = tf.nn.relu6):
	with tf.name_scope(name):
		f = tf.Variable(tf.truncated_normal(fsize, stddev = 0.1), name = "filter")
		conv = tf.nn.conv2d(inputs, f, stride, padding, dilations = [rate, rate, 1, 1], name = "atrous convolution")
		if use_bn:
			mean, var = tf.nn.moments(conv, axes = [0, 1, 2])
			offset = tf.Variable(tf.zeros(conv.get_shape()), name = "offset")
			scale = tf.Variable(tf.ones(conv.get_shape()), name = "scale")
			conv = tf.nn.batch_normalization(conv, mean, var, offset, scale, tf.constant(1e-3), "batch_normalization")
		if activate is not None:
			return activate(conv, name = "activation")
		else:
			return conv

def dense_layer(inputs, out, name, use_bias = True):
	with tf.name_scope(name):
		weight = tf.Variable(tf.truncated_normal([inputs.get_shape()[-1], out], stddev = 0.1), name = "weights")
		dense = tf.matmul(inputs, weight)
		if use_bias:
			bias = tf.Variable(tf.truncated_normal([out], stddev = 0.1), name = "bias")
			dense = tf.nn.bias_add(dense, bias)
		return dense

def residue_block(inputs, depth, channel_out, name, rate = 1, expand_dim = False):
	if expand_dim:
		stride = [1, 2, 2, 1]
	else:
		stride = [1, 1, 1, 1]
	chann = inputs.get_shape()[-1]
	with tf.name_scope(name):
		down = conv_layer(inputs, [1, 1, chann, depth], "downsample", use_bn = False)
		conv = conv_layer(down, [3, 3, depth, depth], "convolution", rate = rate, stride = stride)
		up = conv_layer(conv, [1, 1, depth, channel_out], "up sample", use_bn = False)
		if expand_dim:
			shortcut = conv_layer(inputs, [1, 1, chann, channel_out], "shortcut", stride, use_bn = False)
		else
			shortcut = inputs
		return tf.nn.relu6(up + shortcut)

def resnet(input, size, rate, dense_out = None):
	chann = int(input.get_shape()[-1])
	conv = conv_layer(input, [7, 7, chann, 64], "conv1")
	conv = tf.nn.max_pool(input, [1, 3, 3, 1], [1, 2, 2, 1], "SAME", name = "max_pool_1")
	chann = 64
	for i, s in enumerate(size):
		conv = residue_block(conv, chann, chann * 4, "conv%d_1"%(i+2), rate = rate, expand_dim = (i != 0))
		for j in range(1, s):
			conv = residue_block(conv, chann, chann * 4, "conv%d_%d"(i+2, j+1), rate = rate)
		chann *= 2
	if dense_out is not None:
		fc = tf.reduce_mean(conv, axis = [1, 2])
		fc = tf.dense_layer(conv, dense_out, "dense")
		return conv, fc
	else:
		return conv

def resnet_50(input, rate, dense_out = None):
	return resnet(input, [3, 4, 6, 3], rate, dense_out)

def resnet_101(input, rate, dense_out = None):
	return resnet(input, [3, 4, 23, 3], rate, dense_out)
	