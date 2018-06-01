import deeplab
import tensorflow as tf
from summary import summarizer
batch_size = 16
num_classes = 80

def main(train_type='Resnet', restore=False, maxiter=10, test=False):
	_imgs = tf.placeholder(tf.float32, [None, None, None, 3])
	_labels = tf.placeholder(tf.float32, [None, num_classes])
	_gt = tf.placeholder(tf.float32, [None, None, None, num_classes])

	_deeplab = deeplab.deeplab_v3_plus(_imgs, [128, 64], [48, num_classes], num_classes)
	res_out = _deeplab.get_dense()
	res_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(_labels), logits=res_out, name='res_loss')
	res_mean_loss = tf.reduce_mean(res_loss)
	res_acc = tf.reduce_mean(
		tf.cast(
			tf.equal(
				tf.argmax(res_out, axis = -1),
				tf.argmax(_labels, axis = -1)
			),
			tf.float32
		)
	)
	res_op = tf.train.AdamOptimizer().minimize(res_loss)

	pred_out = _deeplab.get_pred()
	pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels=tf.reshape(tf.stop_gradient(_gt), [-1, num_classes]),
		logits=tf.reshape(pred_out, [-1, num_classes]),
		name='pred_loss')
	pred_mean_loss = tf.reduce_mean(pred_loss)
	pred_op = tf.train.AdamOptimizer().minimize(pred_loss)
	summary = summarizer('./dummy.csv', ['a', 'b', 'c'], 10, restore = False)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			if i % 2 == 0:
				summary(a = 1, b = 0.5 + (i * 0.16) - int(0.5 + (i * 0.16)))
			else:
				summary.summary(a = 1)

main()
