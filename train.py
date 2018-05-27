from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from PythonAPI.pycocotools.coco import COCO
batch_size = 16
num_classes = 80

def main(train_type='Resnet', restore=False, maxiter=10, test=False):
	train_annFile = './annotations/instances_train2017.json'
	val_annFile = './annotations/instances_val2017.json'

	train_coco = COCO(train_annFile)
	val_coco = COCO(val_annFile)
	res_train_dataset, deep_train_dataset = prepare_dataset(train_coco)
	res_val_dataset, deep_val_dataset = prepare_dataset(val_coco)

	res_train_iter = res_train_dataset.make_initializable_iterator()
	deep_train_iter = deep_train_dataset.make_initializable_iterator()
	res_val_iter = res_val_dataset.make_initializable_iterator()
	deep_val_iter = deep_val_dataset.make_initializable_iterator()
	next_res_train = res_train_iter.get_next()
	next_deep_train = deep_train_iter.get_next()
	next_res_val = res_val_iter.get_next()
	next_deep_val = deep_val_iter.get_next()

	# _imgs = tf.placeholder(tf.float32, [batch_size, None, None, 3])
	# _labels = tf.placeholder(tf.float32, [batch_size, num_classes])
	# _gt = tf.placeholder(tf.float32, [batch_size, None, None, num_classes])
	training = tf.placeholder(tf.bool)
	res_end = tf.placeholder(tf.bool)
	_imgs, _y = tf.cond(res_end,
		lambda: tf.cond(training, lambda: next_res_train, lambda: next_res_val),
		lambda: tf.cond(training, lambda: next_deep_train, lambda: next_deep_val)
		)

	_deeplab = deeplab.deeplab_v3_plus(_imgs, [128, 64], [48, num_classes], num_classes)
	res_out = _deeplab.get_dense()
	res_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=_y, logits=res_out, name='res_loss')
	res_mean_loss = tf.reduce_mean(res_loss)
	res_op = tf.train.AdamOptimizer().minimize(res_loss)

	pred_out = _deeplab.get_pred()
	pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(_y, [-1, num_classes]), logits=tf.reshape(pred_out, [-1, num_classes]), name='pred_loss')
	pred_mean_loss = tf.reduce_mean(pred_loss)
	pred_op = tf.train.AdamOptimizer().minimize(pred_loss)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		if restore:
			saver.restore(sess, "./model/model.ckpt")
			print('restored')
		else:
			sess.run(tf.global_variables_initializer())
			print('initial done')

		if train_type == 'Resnet':
			for i in list(range(maxiter)):
				cnt = 0
				total_loss = 0.0
				sess.run(res_train_iter.initializer)
				while True:
					try:
						_, _loss = sess.run([res_op, res_mean_loss], feed_dict={training: True, res_end: True})
						total_loss += _loss
						cnt += 1
						if cnt % 25 == 0:
							print(cnt * 16, total_loss / 25)
							total_loss = 0.0
					except tf.errors.OutOfRangeError:
						break
				print('epoc %d done' % (i + 1))

				cnt = 0
				total_loss = 0.0
				sess.run(res_val_iter.initializer)
				while True:
					try:
						# imgs, labels = sess.run(next_res_val)
						_, _loss = sess.run([res_op, res_mean_loss], feed_dict={training: False, res_end: True})
						total_loss += _loss
						cnt += 1
						if cnt % 25 == 0:
							print(cnt * 16, total_loss / 25)
							total_loss = 0.0
					except tf.errors.OutOfRangeError:
						break

				saver.save(sess, "./model/model.ckpt")
				saver.save(sess, "./model/model_res_%d.ckpt" % i)
				print('model saved')
		elif train_type == 'Deeplab':
			for i in list(range(maxiter)):
				cnt = 0
				total_loss = 0.0
				sess.run(deep_train_iter.initializer)
				while True:
					try:
						#imgs, gt = sess.run(next_deep_train)
						_, _loss = sess.run([res_op, res_mean_loss], feed_dict={_imgs: imgs, _gt: gt})
						total_loss += _loss
						cnt += 1
						if cnt % 25 == 0:
							print(cnt * 16, total_loss / 25)
							total_loss = 0.0
					except tf.errors.OutOfRangeError:
						break
				print('epoc %d done' % (i + 1))

				cnt = 0
				total_loss = 0.0
				sess.run(deep_val_iter.initializer)
				while True:
					try:
						#imgs, gt = sess.run(next_deep_val)
						_, _loss = sess.run([res_op, res_mean_loss], feed_dict={_imgs: imgs, _gt: gt})
						total_loss += _loss
						cnt += 1
						if cnt % 25 == 0:
							print(cnt * 16, total_loss / 25)
							total_loss = 0.0
					except tf.errors.OutOfRangeError:
						break

				saver.save(sess, "./model/model.ckpt")
				saver.save(sess, "./model/model_deep_%d.ckpt" % i)
				print('model saved')
		else:
			pass

if __name__ == '__main__':
	main('Resnet')
