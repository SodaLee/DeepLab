import plot
import json
from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from PythonAPI.pycocotools.coco import COCO
from summary import summarizer
import os
import argparse
batch_size = 24
num_classes = 81
log_dir = "./log"
model_path = "./model/model.ckpt"

def main(train_type='Resnet', restore=False, maxiter=10, test=False):
	train_annFile = './annotations/instances_train2017.json'
	val_annFile = './annotations/instances_val2017.json'
	with open('PythonAPI/cat_dict.json', 'r') as f:
		cat_dict = json.load(f)

	train_coco = COCO(train_annFile)
	val_coco = COCO(val_annFile)
	train_key = []
	val_key = []
	for i in list(range(num_classes-1)):
		train_key.append(train_coco.cats[cat_dict['c2id'][str(i)]]['name'])
		val_key.append(val_coco.cats[cat_dict['c2id'][str(i)]]['name'])
	train_key.append('background')
	val_key.append('background')
	train_set, train_len = prepare_dataset(train_coco, batch_size, [224, 224])
	val_set, val_len = prepare_dataset(val_coco, batch_size, [224, 224])

	handle = tf.placeholder(tf.string, [])
	iterator = tf.data.Iterator.from_string_handle(
		handle,
		(tf.float32, tf.float32, tf.float32),
		((None, None, None, 3), (None, num_classes), (None, None, None, num_classes))
	)
	imgs, labels, gt = iterator.get_next()

	train_iter = train_set.make_initializable_iterator()
	val_iter = val_set.make_initializable_iterator()
	train_handle = train_iter.string_handle()
	val_handle = val_iter.string_handle()

	resnet_step = tf.Variable(0, dtype = tf.int64, name = "resnet_step", trainable = False)
	deep_step = tf.Variable(0, dtype = tf.int64, name = "deep_step", trainable = False)

	_deeplab = deeplab.deeplab_v3_plus(imgs, [128, 64], [48, num_classes], num_classes)
	res_out = _deeplab.get_dense()
	res_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=res_out, name='res_loss')
	res_mean_loss = tf.reduce_mean(res_loss)
	res_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(res_loss, global_step = resnet_step)

	pred_out = _deeplab.get_pred()
	pred_softmax = tf.nn.softmax(pred_out)
	pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels=tf.stop_gradient(gt),
		logits=pred_out,
		name='pred_loss')
	pred_mean_loss = tf.reduce_mean(pred_loss)
	pred_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(pred_loss, global_step = deep_step)
	pred_acc = tf.reduce_mean(tf.cast(tf.argmax(pred_out, -1) == tf.argmax(gt, -1), tf.float32))

	reader = pywrap_tensorflow.NewCheckpointReader(model_path)
	restore_dict = dict()
	for v in tf.get_collection(tf.GraphKeys.VARIABLES):
		tname = v.name.split(':')[0]
		if reader.has_tensor(tname):
			restore_dict[tname] = v

	saver = tf.train.Saver()
	restorer = tf.train.Saver(restore_dict)

	summary = summarizer(
		os.path.join(log_dir, 'log%s.csv'%train_type),
		['step', 'res_train_loss', 'res_val_loss'] if train_type == 'Resnet' else ['step', 'deep_train_loss', 'deep_val_loss'],
		25, restore = restore
	)

	config = tf.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess:
		if restore:
			sess.run(tf.global_variables_initializer())
			restorer.restore(sess, model_path)
			print('restored')
		else:
			sess.run(tf.global_variables_initializer())
			print('initial done')
		
		train_handle = sess.run(train_handle)
		val_handle = sess.run(val_handle)

		if test:
			sess.run(val_iter.initializer)
			cnt = 0
			for epoc in range(100):
				pred = sess.run(pred_softmax, feed_dict = {handle: val_handle})
				print('iter%d' % epoc)
				for i in range(img.shape[0]):
					plot.draw_raw_image(img[i][:,:,::-1], "./test/img_%d_raw.jpg"%cnt)
					plot.draw_image(pred[i], "./test/img_%d_pred.jpg"%cnt)
					cnt += 1
			return


		sess.run([train_iter.initializer, val_iter.initializer])
		if train_type == 'Resnet':
			cnt = 0
			save_cnt = 0
			epoc = 0
			while epoc < maxiter:
				_, _loss = sess.run([res_op, res_mean_loss], feed_dict={handle: train_handle})
				cnt += batch_size
				save_cnt += 1
				if summary.step == summary.steps - 1:
					print("%d/%d %f"%(cnt, train_len, cnt / train_len))
					_valloss = sess.run(res_mean_loss, feed_dict={handle: val_handle})
					summary.summary(res_train_loss = _loss, res_val_loss = _valloss, step = sess.run(resnet_step))
				else:
					summary.summary(res_train_loss = _loss)
				if cnt >= train_len:
					epoc += 1
					print('epoc %d done' % epoc)
					cnt -= train_len

					saver.save(sess, model_path)
					print('model saved')
					save_cnt = 0
				elif save_cnt >= 50:
					save_cnt = 0
					saver.save(sess, model_path)
					print('model saved')
				else:
					pass

		elif train_type == 'Deep':
			cnt = 0
			save_cnt = 0
			epoc = 0
			while epoc < maxiter:
				_, _loss = sess.run([pred_op, pred_mean_loss], feed_dict={handle: train_handle})
				cnt += batch_size
				save_cnt += 1
				if summary.step == summary.steps - 1:
					print("%d/%d %f"%(cnt, train_len, cnt / train_len))
					_valloss = sess.run(pred_mean_loss, feed_dict={handle: val_handle})
					summary.summary(deep_train_loss = _loss, deep_val_loss = _valloss, step = sess.run(deep_step))
				else:
					summary.summary(deep_train_loss = _loss)
				if cnt >= train_len:
					epoc += 1
					print('epoc %d done' % epoc)
					cnt -= train_len

					saver.save(sess, model_path)
					print('model saved')
					save_cnt = 0
				elif save_cnt >= 50:
					save_cnt = 0
					saver.save(sess, model_path)
					print('model saved')
				else:
					pass
		else:
			assert False, "unknown training type %s" % train_type

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-g", "--gpu", help = "specify which gpu to use", default = "0")
	parser.add_argument("-t", "--type",
		help = "specify the type of training/test",
		choices = ["Resnet", "Deep"],
		default = "Resnet")
	parser.add_argument("-l", "--log", help = "directory name of log files", default = "./log")
	parser.add_argument("-r", "--restore", help = "restore from checkpoint", action = "store_true")
	parser.add_argument("-i", "--maxiter", help = "maximum epoc", type = int, default = 10)
	parser.add_argument("-c", "--checkpoint", help = "checkpoint path", default = "./model/model.ckpt")
	parser.add_argument("--test", help = "test deeplab", action = "store_true")

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	log_dir = args.log
	model_path = args.checkpoint
	main(args.type, restore = args.restore, maxiter = args.maxiter, test = args.test)
