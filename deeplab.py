import tensorflow as tf
from resnet import resnet_50, resnet_101

def ASPP(x, channel_out, dilations, name):
	out = []
	with tf.name_scope(name):
		for i, d in enumerate(dilations):
			out.append(conv_layer(x, 3, channel_out, "conv%d"%(i+1), rate = d))
		out.append(conv_layer(x, 1, channel_out, "conv%d"%(len(dilations)+1)))
		return tf.concat(out, -1)
		
def upsample(inputs, scale, name = None):
	with tf.name_scope(name):
		channel = inputs.shape()[-1]
		f = tf.Variable(tf.truncated_normal([scale+1, scale+1, channel, channel], stddev = 0.1), "filter")
		deconv = tf.nn.conv2d_transpose(inputs, f, tf.shape(inputs), [1,scale,scale,1], padding = "SAME", name = "deconvolution")
		return deconv

def encoder(x, aspp_channel1, aspp_channel2, name):
	with tf.name_scope(name):
		dcnn = resnet_50(x)
		aspp = ASPP(dcnn, aspp_channel1, [6, 12, 18, 24], "ASPP")
		aspp = conv_layer(aspp, 1, aspp_channel2, "conv")
		return dcnn, aspp


def decoder(dcnn, aspp, channels, name):
	with name_scope(name):
		conv1 = conv_layer(dcnn, 1, channels[0], "conv1")
		conv = tf.concat([conv1, upsample(dcnn, 2, "upsample1")], -1)
		for i, c in enumerate(channels, 1):
			conv = conv_layer(conv, 3, c, "conv%d"%(i+1))
		conv = upsample(conv, 2, "upsample2")
		return conv

def deeplab_v3_plus(x, channels, aspp_channels):
	dcnn, aspp = encoder(x, aspp_channels[0], aspp_channels[1], "encoder")
	return decoder(dcnn, aspp, channels, "decoder")
