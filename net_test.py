import deeplab
import tensorflow as tf
from summary import summarizer
import numpy as np
from utli import crf_rnn
batch_size = 2
num_classes = 20

def main(train_type='Resnet', restore=False, maxiter=10, test=False):
	train_data = (np.random.rand(10, 64, 64, 3), np.random.rand(10, 20), np.random.rand(10, 64, 64, 20))
	val_data = (np.random.rand(2, 64, 64, 3), np.random.rand(2, 20), np.random.rand(2, 64, 64, 20))
	map_fn = lambda x: x.astype(np.float32)
	train_data = tuple(map(map_fn, train_data))
	val_data = tuple(map(map_fn, val_data))

	train_set = tf.data.Dataset.from_tensor_slices(train_data)
	val_set = tf.data.Dataset.from_tensor_slices(val_data)
	train_set = train_set.batch(batch_size).repeat()
	val_set = val_set.batch(batch_size).repeat()

	handle = tf.placeholder(tf.string, shape = [])
	iterator = tf.data.Iterator.from_string_handle(
		handle,
		(tf.float32, tf.float32, tf.float32),
		((None, None, None, 3), (None, 20), (None, None, None, 20))
	)
	imgs, labels, gt = iterator.get_next()

	train_iter = train_set.make_initializable_iterator()
	val_iter = val_set.make_initializable_iterator()
	train_handle = train_iter.string_handle()
	val_handle = val_iter.string_handle()

	resnet_step = tf.Variable(0, dtype = tf.int64, name = "resnet_step", trainable = False)
	deep_step = tf.Variable(0, dtype = tf.int64, name = "deep_step", trainable = False)
	crf_step = tf.Variable(0, dtype = tf.int64, name = "crf_step", trainable = False)
	crf_separate = tf.placeholder(tf.bool, shape = [])

	_deeplab = deeplab.deeplab_v3_plus(imgs, [128, 64], [48, num_classes], num_classes)
	res_out = _deeplab.get_dense()
	res_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=res_out, name='res_loss')
	res_mean_loss = tf.reduce_mean(res_loss)
	res_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(res_loss, global_step = resnet_step)

	pred_out = _deeplab.get_pred()
	pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels=tf.stop_gradient(gt),
		logits=pred_out,
		name='pred_loss')
	pred_mean_loss = tf.reduce_mean(pred_loss)
	pred_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(pred_loss, global_step = deep_step)
	pred_acc = tf.reduce_mean(tf.cast(tf.argmax(pred_out, -1) == tf.argmax(gt, -1), tf.float32))

	crf_in = tf.cond(crf_separate, lambda: tf.stop_gradient(pred_out), lambda: pred_out)
	crf_out = crf_rnn(crf_in, imgs, tf.constant([1., 1., 1., .5, .5], dtype = tf.float32), 5, "crf_rnn")
	crf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels = tf.stop_gradient(gt),
		logits = crf_out,
		name = "crf_loss")
	crf_mean_loss = tf.reduce_mean(crf_loss)
	crf_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(crf_loss, global_step = crf_step)

	summary = summarizer(
		'./log/test.csv',
		['a', 'b', 'c'],
		25, True
	)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_handle = sess.run(train_handle)
		val_handle = sess.run(val_handle)

		for i in range(100):
			if i % 2 == 0:
				summary(a = 1, b = 0.5 + (i * 0.16) - int(0.5 + (i * 0.16)))
			else:
				summary.summary(a = 1)

		sess.run(train_iter.initializer)
		sess.run(val_iter.initializer)
		
		for i in range(5):
			sess.run(res_op, feed_dict = {handle: train_handle})
			print(i)
		print(sess.run(res_mean_loss, feed_dict = {handle: val_handle}))
		for i in range(5):
			sess.run(pred_op, feed_dict = {handle: train_handle})
			print(i)
		print(sess.run(pred_acc, feed_dict = {handle: val_handle}))
		for i in range(5):
			sess.run(crf_op, feed_dict = {handle: train_handle, crf_separate: True})
			print(i)
		print(sess.run(crf_mean_loss, feed_dict = {handle: val_handle, crf_separate: True}))

main()
