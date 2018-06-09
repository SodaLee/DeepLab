from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import tensorflow as tf
import itertools
from urllib.request import urlretrieve
import os
import cv2
import json

def prepare_dataset(coco, batch_size, img_size = [128, 128]):

	def _parse_fn(imgId):
		img = coco.imgs[imgId]
		fname = os.path.join(imgDir, img['file_name'])
		if not os.path.exists(fname):
			urlretrieve(img['coco_url'], fname)
		img_decoded = cv2.imread(fname)

		annLists = [coco.imgToAnns[imgId]]
		anns = list(itertools.chain.from_iterable(annLists))
		masks = np.zeros([num_classes, img_decoded.shape[0], img_decoded.shape[1]])
		for ann in anns:
			mask = coco.annToMask(ann)
			masks[cat_dict['id2c'][str(ann['category_id'])]] += mask
		masks_sum = np.sum(masks, axis = 0, keepdims = True)
		masks = np.where(
			masks_sum > 0,
			masks / masks_sum,
			np.concatenate([
				np.zeros([num_classes - 1, img_decoded.shape[0], img_decoded.shape[1]]),
				np.ones([1, img_decoded.shape[0], img_decoded.shape[1]])
				],
				axis = 0)
		).astype(np.float32)

		labels = np.sum(masks, axis = (1, 2))
		labels /= np.sum(labels)

		masks = masks.transpose(1, 2, 0)

		return img_decoded.astype(np.float32) / 255, labels, masks
	
	def _resize_fn(img, labels, masks):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, img_size)
		labels.set_shape([num_classes])
		masks.set_shape([None, None, num_classes])
		masks = tf.image.resize_images(masks, img_size)
		return img, labels, masks

	imgDir = 'data/imgs'
	annDir = 'data/anns'
	imgIds = coco.getImgIds()
	with open('PythonAPI/cat_dict.json', 'r') as f:
		cat_dict = json.load(f)
	num_classes = 81

	dataset = tf.data.Dataset.from_tensor_slices(imgIds)
	dataset = dataset.map(map_func = lambda imgId: tf.py_func(_parse_fn, [imgId], [tf.float32, tf.float32, tf.float32], name = "parse_data"), num_parallel_calls = 8)
	dataset = dataset.map(map_func = _resize_fn, num_parallel_calls = 8)
	dataset = dataset.batch(batch_size).repeat()
	dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/device:GPU:0"))

	return dataset, len(imgIds)
