import tensorflow as tf

def batch_norm(inputs, name, _global = False):
	with tf.name_scope(name):
		if _global:
			mean, var = tf.nn.moments(inputs, axes = [0, 1, 2])
			offset = tf.Variable(tf.zeros(inputs.get_shape()[-1]), name = "offset")
			scale = tf.Variable(tf.ones(inputs.get_shape()[-1]), name = "scale")
		else:
			mean, var = tf.nn.moments(inputs, axes = [0])
			offset = tf.Variable(tf.zeros(inputs.get_shape()[1:]), name = "offset")
			scale = tf.Variable(tf.ones(inputs.get_shape()[1:]), name = "scale")
		bn = tf.nn.batch_normalization(
			inputs, mean, var, offset, scale, tf.constant(1e-3), "batch_normalization")
		return bn

def conv_layer(inputs, fsize, channel_out, name, stride = [1,1,1,1],
	rate = 1, padding = "SAME", use_bn = True, activate = tf.nn.relu6):
	filter_size = [fsize, fsize, inputs.get_shape()[-1], channel_out]
	filter_size = tf.TensorShape(filter_size)
	with tf.name_scope(name):
		f = tf.Variable(tf.truncated_normal(filter_size, stddev = 0.1), name = "filter")
		convname = "atrous_convolution" if rate != 1 else "convolution"
		conv = tf.nn.conv2d(inputs, f, stride, padding, dilations = [1, 1, rate, rate], name = convname)
		if use_bn:
			conv = batch_norm(conv, "batch_norm", True)
		if activate is not None:
			return activate(conv, name = "activation")
		else:
			return conv

def dense_layer(inputs, out, name, use_bias = True):
	with tf.name_scope(name):
		wshape = [inputs.get_shape()[-1], out]
		wshape = tf.TensorShape(wshape)
		weight = tf.Variable(tf.truncated_normal(wshape, stddev = 0.1), name = "weights")
		dense = tf.matmul(inputs, weight)
		if use_bias:
			bias = tf.Variable(tf.truncated_normal([out], stddev = 0.1), name = "bias")
			dense = tf.nn.bias_add(dense, bias)
		return dense