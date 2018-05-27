import tensorflow as tf

def conv_layer(inputs, fsize, channel_out, name, stride = [1,1,1,1],
	rate = 1, padding = "SAME", use_bn = True, activate = tf.nn.relu6):
	filter_size = [fsize, fsize, inputs.get_shape()[-1], channel_out]
	filter_size = tf.TensorShape(filter_size)
	with tf.name_scope(name):
		f = tf.Variable(tf.truncated_normal(filter_size, stddev = 0.1), name = "filter")
		convname = "atrous_convolution" if rate != 1 else "convolution"
		conv = tf.nn.conv2d(inputs, f, stride, padding, dilations = [1, 1, rate, rate], name = convname)
		if use_bn:
			mean, var = tf.nn.moments(conv, axes = [0])
			offset = tf.Variable(tf.zeros(conv.get_shape()[1:]), name = "offset")
			scale = tf.Variable(tf.ones(conv.get_shape()[1:]), name = "scale")
			conv = tf.nn.batch_normalization(conv, mean, var, offset, scale, tf.constant(1e-3), "batch_normalization")
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

def residue_block(inputs, depth, channel_out, name, half_size = False):
	with tf.name_scope(name):
		if half_size:
			stride = [1,2,2,1]
			shortcut = conv_layer(inputs, 1, channel_out, "shortcut", stride = stride)
		else:
			stride = [1,1,1,1]
			if inputs.get_shape().as_list()[-1] != channel_out:
				shortcut = conv_layer(inputs, 1, channel_out, "shortcut", stride = stride)
			else:
				shortcut = inputs
		
		down = conv_layer(inputs, 1, depth, "in", stride = stride)
		conv = conv_layer(down, 3, depth, "convolution")
		up = conv_layer(conv, 1, channel_out, "out", activate = None)

		return tf.nn.relu6(up + shortcut, name = "activation")

def atrous_residue_block(inputs, depth, channel_out, rate, name):
	with tf.name_scope(name):
		if inputs.get_shape().as_list()[-1] != channel_out:
			shortcut = conv_layer(inputs, 1, channel_out, "shortcut", stride = [1,1,1,1])
		else:
			shortcut = inputs

		down = conv_layer(inputs, 1, depth, "in")
		conv = conv_layer(down, 3, depth, "convolution", rate = rate)
		up = conv_layer(conv, 1, channel_out, "out", activate = None)
		return tf.nn.relu6(up + shortcut, name = "activation")

def resnet(x, nconvs, name):
	assert nconvs[0] == 1, "conv1 should only contain one convolution layer"
	with tf.name_scope(name):
		conv1 = conv_layer(x, 7, 64, "conv1", stride = [1,2,2,1])
		pool1 = tf.nn.max_pool(conv1, [1,3,3,1], [1,2,2,1], padding = "SAME", name = "pool1")
		conv = pool1
		depth = 64
		for i in range(1, len(nconvs) - 2):
			for j in range(nconvs[i]):
				conv = residue_block(conv, depth, depth * 4, "conv%d_%d"%(i+1,j+1), half_size = (i > 1 and j == 0))
			depth *= 2
			if i == 1:
				llf = conv

		for i in range(len(nconvs) - 2, len(nconvs)):
			for j in range(nconvs[i]):
				conv = atrous_residue_block(conv, depth, depth * 4, 2 * (i - len(nconvs) + 3), "conv%d_%d"%(i+1,j+1))
			depth *= 2
		return conv, llf

def resnet_50(x):
	return resnet(x, [1, 3, 4, 6, 3], "ResNet-50")

def resnet_101(x, dense_out = 0):
	return resnet(x, [1, 3, 4, 23, 3], "ResNet-101")
