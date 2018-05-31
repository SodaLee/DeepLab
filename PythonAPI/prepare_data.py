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

	def _parse_res(imgId):
		img = coco.imgs[imgId]
		fname = os.path.join(imgDir, img['file_name'])
		if not os.path.exists(fname):
			urlretrieve(img['coco_url'], fname)
		img_decoded = cv2.imread(fname)

		annLists = [coco.imgToAnns[imgId]]
		anns = list(itertools.chain.from_iterable(annLists))
		label = np.zeros(num_classes, np.float32)
		for ann in anns:
			label[cat_dict['id2c'][str(ann['category_id'])]] = 1
		label_sum = np.sum(label)
		if label_sum > 0:
			label /= label_sum
		else:
			label[num_classes-1] = 1

		return img_decoded.astype(np.float32) / 255, label

	def _parse_deep(imgId):
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
		)

		return img_decoded.astype(np.float32) / 255, masks.transpose(1, 2, 0).astype(np.float32)

	def _resize_res(img, label):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, img_size)
		label.set_shape([num_classes])
		return img, label

	def _resize_deep(img, masks):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, img_size)
		masks.set_shape([None, None, num_classes])
		masks = tf.image.resize_images(masks, img_size)
		return img, masks

	imgDir = 'data/imgs'
	annDir = 'data/anns'
	imgIds = coco.getImgIds()
	with open('PythonAPI/cat_dict.json', 'r') as f:
		cat_dict = json.load(f)
	num_classes = 81

	dataset = tf.data.Dataset.from_tensor_slices(imgIds)

	res_dataset = dataset.map(lambda imgId: tf.py_func(_parse_res, [imgId], [tf.float32, tf.float32], name='parse_res'))
	res_dataset = res_dataset.map(_resize_res)
	res_dataset = res_dataset.batch(batch_size).repeat()
	
	deep_dataset = dataset.map(lambda imgId: tf.py_func(_parse_deep, [imgId], [tf.float32, tf.float32], name='parse_deep'))
	deep_dataset = deep_dataset.map(_resize_deep)
	deep_dataset = deep_dataset.batch(batch_size).repeat()

	return res_dataset, deep_dataset, len(imgIds)
