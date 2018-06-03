from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from PythonAPI.pycocotools.coco import COCO
from summary import summarizer
import os
import argparse
batch_size = 16
num_classes = 81
log_dir = "./log"
model_path = "./model/model.ckpt"

def main(train_type='Resnet', restore=False, maxiter=10, test=False):
	train_annFile = './annotations/instances_train2017.json'
	val_annFile = './annotations/instances_val2017.json'

	train_coco = COCO(train_annFile)
	val_coco = COCO(val_annFile)
	res_train_dataset, deep_train_dataset, train_len = prepare_dataset(train_coco, batch_size, [224, 224])
	res_val_dataset, deep_val_dataset, val_len = prepare_dataset(val_coco, batch_size, [224, 224])

	dataset = [[res_train_dataset, res_val_dataset], [deep_train_dataset, deep_val_dataset]]
	iterator = list(map(lambda x: list(map(lambda y: y.make_initializable_iterator(), x)), dataset))
	pairs = list(map(lambda x: list(map(lambda y: y.get_next(), x)), iterator))
	initializer = list(map(lambda x: list(map(lambda y: y.initializer, x)), iterator))

	_imgs = tf.placeholder(tf.float32, [None, None, None, 3])
	_labels = tf.placeholder(tf.float32, [None, num_classes])
	_gt = tf.placeholder(tf.float32, [None, None, None, num_classes])

	resnet_step = tf.Variable(0, dtype = tf.int64, name = "resnet_step", trainable = False)
	deep_step = tf.Variable(0, dtype = tf.int64, name = "deep_step", trainable = False)

	_deeplab = deeplab.deeplab_v3_plus(_imgs, [128, 64], [48, num_classes], num_classes)
	res_out = _deeplab.get_dense()
	res_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(_labels), logits=res_out, name='res_loss')
	res_mean_loss = tf.reduce_mean(res_loss)
	res_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(res_loss, global_step = resnet_step)

	pred_out = _deeplab.get_pred()
	pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels=tf.reshape(tf.stop_gradient(_gt), [-1, num_classes]),
		logits=tf.reshape(pred_out, [-1, num_classes]),
		name='pred_loss')
	pred_mean_loss = tf.reduce_mean(pred_loss)
	pred_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(pred_loss, global_step = deep_step)

	if os.path.isdir(model_path):
		ckpt = tf.train.get_checkpoint_state(model_path)
	else:
		ckpt = tf.train.get_checkpoint_state(os.path.split(model_path)[0], os.path.split(model_path)[1])
	reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()
	ckpt_var_list = var_to_shape_map.values()

	saver = tf.train.Saver()
	restorer = tf.train.Saver(ckpt_var_list)

	summary = summarizer(
		os.path.join(log_dir, 'log%s.csv'%train_type),
		['step', 'res_train_loss', 'res_val_loss'] if train_type == 'Resnet' else ['step', 'deep_trian_loss', 'deep_val_loss'],
		25, restore = restore
	)

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess:
		if restore:
			restorer.restore(sess, model_path)
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
				_, _loss = sess.run([res_op, res_mean_loss], feed_dict={_imgs: img, _labels: label})
				cnt += batch_size
				if summary.step == summary.steps - 1:
					print("%d/%d %f"%(cnt, train_len, cnt / train_len))
					img, label = sess.run(pairs[0][1])
					_valloss = sess.run(res_mean_loss, feed_dict={_imgs: img, _labels: label})
					summary.summary(res_train_loss = _loss, res_val_loss = _valloss, step = sess.run(resnet_step))
				else:
					summary.summary(res_train_loss = _loss, step = sess.run(resnet_step))
				if cnt >= train_len:
					epoc += 1
					print('epoc %d done' % epoc)
					cnt -= train_len

					saver.save(sess, model_path)
					print('model saved')

		elif train_type == 'Deeplab':
			sess.run(initializer[1])
			cnt = 0
			epoc = 0
			while epoc < maxiter:
				img, gt = sess.run(pairs[1][0])
				_, _loss = sess.run([deep_op, deep_mean_loss], feed_dict={_imgs: img, _gt: gt})
				cnt += batch_size
				if summary.step == summary.steps - 1:
					print("%d/%d %f"%(cnt, train_len, cnt / train_len))
					img, gt = sess.run(pairs[1][1])
					_valloss = sess.run(deep_mean_loss, feed_dict={_imgs: img, _gt: gt})
					summary.summary(deep_train_loss = _loss, deep_val_loss = _valloss, step = sess.run(deep_step))
				else:
					summary.summary(deep_train_loss = _loss, step = sess.run(deep_step))
				if cnt >= train_len:
					epoc += 1
					print('epoc %d done' % epoc)
					cnt -= train_len

					saver.save(sess, model_path)
					print('model saved')
		else:
			assert False, "unknown training type"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-g", "--gpu", help = "specify which gpu to use", default = "0")
	parser.add_argument("-t", "--type",
		help = "specify the type of training",
		choices = ["Resnet", "Deep"],
		default = "Resnet")
	parser.add_argument("-l", "--log", help = "directory name of log files", default = "./log")
	parser.add_argument("-r", "--restore", help = "restore from checkpoint", action = "store_true")
	parser.add_argument("-i", "--maxiter", help = "maximum epoc", type = int, default = 10)
	parser.add_argument("-c", "--checkpoint", help = "checkpoint path", default = "./model/model.ckpt")

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	log_dir = args.log
	model_path = args.checkpoint
	main(args.type, restore = args.restore, maxiter = args.maxiter, test = False)
