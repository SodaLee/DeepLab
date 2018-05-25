import tensorflow as tf
from resnet import *

class deeplab_v3_plus(object):
	def __init__(self, x, aspp_channels, decoder_channels, dense_out = 0):
		self.dense_out = dense_out
		self.dcnn, self.aspp = encoder(x, aspp_channels[0], aspp_channels[1], "encoder")
		self.net = decoder(dcnn, aspp, decoder_channels, "decoder")

	def _ASPP(self, x, channel_out, dilations, name):
		out = []
		with tf.name_scope(name):
			for i, d in enumerate(dilations):
				out.append(conv_layer(x, 3, channel_out, "conv%d"%(i+1), rate = d))
			out.append(conv_layer(x, 1, channel_out, "conv%d"%(len(dilations)+1)))
			return tf.concat(out, -1)
			
	def _upsample(self, inputs, scale, name):
		with tf.name_scope(name):
			channel = inputs.shape()[-1]
			f = tf.Variable(tf.truncated_normal([scale+1, scale+1, channel, channel], stddev = 0.1), "filter")
			deconv = tf.nn.conv2d_transpose(inputs, f, tf.shape(inputs), [1,scale,scale,1], padding = "SAME", name = "deconvolution")
			return deconv

	def _encoder(self, x, aspp_channel1, aspp_channel2, name):
		with tf.name_scope(name):
			dcnn = resnet_50(x)
			if self.dense_out != 0:
				self.dense = tf.reduce_mean(dcnn, axis = [1, 2])
				self.dense = denes_layer(self.dense, self.dense_out)
			dcnn = tf.stop_gradient(dcnn)
			aspp = self._ASPP(dcnn, aspp_channel1, [6, 12, 18, 24], "ASPP")
			aspp = conv_layer(aspp, 1, aspp_channel2, "conv")
			return dcnn, aspp


	def _decoder(self, dcnn, aspp, channels, name):
		with name_scope(name):
			conv1 = conv_layer(dcnn, 1, channels[0], "conv1")
			conv = tf.concat([conv1, self._upsample(aspp, 2, "upsample1")], -1)
			for i, c in enumerate(channels, 1):
				conv = conv_layer(conv, 3, c, "conv%d"%(i+1))
			conv = self._upsample(conv, 2, "upsample2")
			return conv

	def get_pred(self):
		return self.net

	def get_dense(self):
		return self.dense
'''
sample:
deeplab = deeplab_v3_plus(x, [256, 256], [48, 80], 100)
net = deeplab.get_pred()
resnet_train = deeplab.get_dense()
...
'''
