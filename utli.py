import tensorflow as tf
import custom

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

def message_passing(unary, raw, kernel, expand_dims = True):
	Q = custom.layer.message_passing(unary, raw, kernel)
	if expand_dims:
		return tf.expand_dims(Q, -1)
	else:
		return Q

def crf_cell(H, U, raw, kernels, name):
	nclass = H.get_shape().as_list()[-1]
	raw = tf.stop_gradient(raw)
	U = tf.stop_gradient(U)
	with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
		if(isinstance(kernels, tf.Tensor) or type(kernels[0]) != list):
			Q = message_passing(H, raw, kernels, False)
		else:
			Qs = []
			for k in kernels:
				Qs.append(message_passing(H, raw, k))
			Q = tf.concat(Qs, -1)
			weights = tf.get_variable("filter_weights", [1, 1, 1, len(kernels), 1], initializer = tf.truncated_normal_initializer(stddev = 0.1))
			Q = tf.nn.conv3d(Q, weights, [1,1,1,1,1], "SAME")
		compati = tf.get_variable("compatibility_matrix", [nclass, nclass], initializer = tf.random_uniform_initializer())
		Q = tf.reshape(tf.matmul(tf.reshape(Q, [-1, nclass]), compati), tf.shape(U))
		Q = U - Q
		return Q

def crf_rnn(unary, raw, kernels, maxiter, name):
	with tf.name_scope(name):
		H = unary
		for i in range(maxiter):
			H = tf.nn.softmax(H)
			H = crf_cell(H, unary, raw, kernels, "crf_cell")
		return H

