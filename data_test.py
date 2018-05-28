from PythonAPI.prepare_data import prepare_dataset
import deeplab
import tensorflow as tf
from PythonAPI.pycocotools.coco import COCO
import cv2
batch_size = 16
num_classes = 80

def main(train_type='Resnet'):
	val_annFile = './annotations/instances_val2017.json'

	val_coco = COCO(val_annFile)
	res_val_dataset, deep_val_dataset = prepare_dataset(val_coco)

	res_val_iter = res_val_dataset.make_initializable_iterator()
	deep_val_iter = deep_val_dataset.make_initializable_iterator()
	next_res_val = res_val_iter.get_next()
	next_deep_val = deep_val_iter.get_next()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('initial done')

		if train_type == 'Resnet':
				cnt = 0
				total_loss = 0.0
				sess.run(res_val_iter.initializer)
				while True:
					try:
						imgs, y = sess.run(next_res_val)
						print(imgs.shape, y.shape)
					except tf.errors.OutOfRangeError:
						break
		else:
			pass

if __name__ == '__main__':
	main('Resnet')