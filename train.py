from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from PythonAPI.pycocotools.coco import COCO
from summary import summarizer
batch_size = 4
num_classes = 81

def main(train_type='Resnet', restore=False, maxiter=10, test=False):
	train_annFile = './annotations/instances_train2017.json'
	val_annFile = './annotations/instances_val2017.json'

	train_coco = COCO(train_annFile)
	val_coco = COCO(val_annFile)
	res_train_dataset, deep_train_dataset, train_len = prepare_dataset(train_coco, batch_size, [128, 128])
	res_val_dataset, deep_val_dataset, val_len = prepare_dataset(val_coco, batch_size, [128, 128])

	dataset = [[res_train_dataset, res_val_dataset], [deep_train_dataset, deep_val_dataset]]
	iterator = list(map(lambda x: list(map(lambda y: y.make_initializable_iterator(), x)), dataset))
	pairs = list(map(lambda x: list(map(lambda y: y.get_next(), x)), iterator))
	initializer = list(map(lambda x: list(map(lambda y: y.initializer, x)), iterator))

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

	saver = tf.train.Saver()
	summary = summarizer(
		'./log/log%s.csv'%train_type,
		['res_train_loss', 'res_train_acc', 'res_val_loss', 'res_val_acc'] if train_type == 'Resnet' else ['deep_trian_loss', 'deep_val_loss'],
		25
	)

	with tf.Session() as sess:
		if restore:
			saver.restore(sess, "./model/model.ckpt")
			print('restored')
		else:
			sess.run(tf.global_variables_initializer())
			print('initial done')

		if train_type == 'Resnet':
			sess.run(initializer[0])
			cnt = 0
			epoc = 0
			while epoc < maxiter:
				img, label = sess.run(pairs[0][0])
				_, _loss, _acc = sess.run([res_op, res_mean_loss, res_acc], feed_dict={_imgs: img, _labels: label})
				if summary.step == summary.steps - 1:
					img, label = sess.run(pairs[0][1])
					_valloss, _valacc = sess.run([res_mean_loss, res_acc], feed_dict={_imgs: img, _labels: label})
					summary.summary(res_train_loss = _loss, res_train_acc = _acc, res_val_loss = _valloss, res_val_acc = _valacc)
				else:
					summary.summary(res_train_loss = _loss, res_train_acc = _acc)
				cnt += batch_size
				if cnt >= train_len:
					epoc += 1
					print('epoc %d done' % (epoc + 1))
					cnt -= train_len

			saver.save(sess, "./model/model.ckpt")
			#saver.save(sess, "./model/model_res_%d.ckpt" % i)
			print('model saved')

		elif train_type == 'Deeplab':
			sess.run(initializer[1])
			cnt = 0
			epoc = 0
			while epoc < maxiter:
				img, gt = sess.run(pairs[1][0])
				_, _loss = sess.run([deep_op, deep_mean_loss], feed_dict={_imgs: img, _gt: gt})
				if summary.step == summary.steps - 1:
					img, gt = sess.run(pairs[1][1])
					_valloss = sess.run([deep_mean_loss], feed_dict={_imgs: img, _gt: gt})
					summary.summary(deep_train_loss = _loss, deep_val_loss = _valloss)
				else:
					summary.summary(deep_train_loss = _loss)
				cnt += batch_size
				if cnt >= train_len:
					epoc += 1
					print('epoc %d done' % (epoc + 1))
					cnt -= train_len

			saver.save(sess, "./model/model.ckpt")
			#saver.save(sess, "./model/model_deep_%d.ckpt" % i)
			print('model saved')
		else:
			pass

if __name__ == '__main__':
	main('Resnet')
