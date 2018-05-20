import tensorflow as tf

def subsample(inputs, factor, scope=None):
	if factor == 1:
		return inputs
	else:
		return tf.nn.max_pool(inputs, [1, 1, 1, 1], [1, factor, factor, 1])

def bottleneck(inputs,
			   depth,
			   depth_bottleneck,
			   stride,
			   unit_rate=1,
			   rate=1,
			   scope=None):
	with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
		depth_in = inputs.get_shape()
		if depth == depth_in:
			shortcut = subsample(inputs, stride, 'shortcut')
		else:
			shortcut = tf.nn.conv2d(inputs, [1, 1, 1, depth], [1, stride, stride, 1], padding='SAME')

		residual = tf.nn.conv2d(inputs, [1, 1, 1, depth_bottleneck], [1, 1, 1, 1], name='conv1')