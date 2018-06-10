import tensorflow as tf
from tensorflow.python.framework import ops
import os

custom_layer = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'cpp', 'MessagePassing.so'))

@ops.RegisterGradient('MessagePassing')
def MessagePassingGrad(op, grad):
	raw = op.inputs[1]
	kernel = op.inputs[2]
	newgrad = custom_layer.MessagePassing(grad, raw, kernel, reverse = True)

	return [newgrad, tf.zeros_like(raw), tf.zeros_like(kernel)]
