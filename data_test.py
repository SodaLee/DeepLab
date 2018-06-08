from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from PythonAPI.pycocotools.coco import COCO
import cv2
import time
batch_size = 16
num_classes = 80

def main(train_type='Resnet'):
	val_annFile = './annotations/instances_val2017.json'

	val_coco = COCO(val_annFile)
	res_val_dataset, deep_val_dataset, num_imgs = prepare_dataset(val_coco, 16)

	res_val_iter = res_val_dataset.make_initializable_iterator()
	deep_val_iter = deep_val_dataset.make_initializable_iterator()
	next_res_val = res_val_iter.get_next()
	next_deep_val = deep_val_iter.get_next()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('initial done')

		tic = time.time()
		if train_type == 'Resnet':
				cnt = 0
				total_loss = 0.0
				sess.run(deep_val_iter.initializer)
				while True:
					try:
						imgs, y = sess.run(next_deep_val)
						cnt += 16
						print('total', cnt, imgs.shape, y.shape)
						if cnt >= 128:
							break
					except tf.errors.OutOfRangeError:
						break
		else:
			pass
		print('Done (t={:0.2f}s)'.format(time.time()- tic))

if __name__ == '__main__':
	main('Resnet')
