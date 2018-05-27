from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import tensorflow as tf
import itertools
from urllib.request import urlretrieve
import os
import cv2
import json

def prepare_dataset(coco):

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

		return img_decoded.astype(np.float32), label

	def _parse_deep(imgId):
		img = coco.imgs[imgId]
		fname = os.path.join(imgDir, img['file_name'])
		if not os.path.exists(fname):
			urlretrieve(img['coco_url'], fname)
		img_decoded = cv2.imread(fname)

		annLists = [coco.imgToAnns[imgId]]
		anns = list(itertools.chain.from_iterable(annLists))
		masks = np.zeros([num_classes, img.shape[0], img.shape[1]])
		for ann in anns:
			mask = coco.annToMask(ann)
			masks[cat_dict['id2c'][str(ann['category_id'])]] += mask

		return img_decoded.astype(np.float32), masks.transpose(1, 2, 0).astype(np.float32)

	def _resize_res(img, label):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, [512, 512])
		label.set_shape([num_classes])
		return img, label

	def _resize_deep(img, masks):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, [512, 512])
		masks.set_shape([None, None, num_classes])
		masks = tf.image.resize_images(masks, [512, 512])
		return img, masks

	imgDir = 'data/imgs'
	annDir = 'data/anns'
	imgIds = coco.getImgIds()
	with open('PythonAPI/cat_dict.json', 'r') as f:
		cat_dict = json.load(f)
	num_classes = 80

	res_dataset = tf.data.Dataset.from_tensor_slices(imgIds)
	res_dataset = res_dataset.map(lambda imgId: tf.py_func(_parse_res, [imgId], [tf.float32, tf.float32], name='parse_res'))
	res_dataset = res_dataset.map(_resize_res)
	res_dataset = res_dataset.batch(16)
	
	deep_dataset = tf.data.Dataset.from_tensor_slices(imgIds)
	deep_dataset = deep_dataset.map(lambda imgId: tf.py_func(_parse_deep, [imgId], [tf.float32, tf.float32], name='parse_deep'))
	deep_dataset = deep_dataset.map(_resize_deep)
	deep_dataset = deep_dataset.batch(16)

	return res_dataset, deep_dataset
