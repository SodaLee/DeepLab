from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from PythonAPI.pycocotools.coco import COCO
import cv2
import time
batch_size = 16
num_classes = 81

def main(train_type='Resnet'):
	val_annFile = './annotations/instances_val2017.json'

	val_coco = COCO(val_annFile)
	val_set, val_len = prepare_dataset(val_coco, batch_size, [224, 224])

	handle = tf.placeholder(tf.string, [])
	iterator = tf.data.Iterator.from_string_handle(
		handle,
		(tf.float32, tf.float32, tf.float32),
		((None, None, None, 3), (None, num_classes), (None, None, None, num_classes))
	)
	imgs, labels, gt = iterator.get_next()

	val_iter = val_set.make_initializable_iterator()
	val_handle = val_iter.string_handle()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		val_handle = sess.run(val_handle)
		print('initial done')

		tic = time.time()
		if train_type == 'Resnet':
				cnt = 0
				total_loss = 0.0
				sess.run(val_iter.initializer)
				while True:
					try:
						img, y = sess.run([imgs, labels], feed_dict={handle: val_handle})
						cnt += 16
						print('total', cnt, img.shape, y.shape)
						if cnt >= 128:
							break
					except tf.errors.OutOfRangeError:
						break
		else:
			pass
		print('Done (t={:0.2f}s)'.format(time.time()- tic))

if __name__ == '__main__':
	main('Resnet')
