from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import tensorflow as tf
from urllib.request import urlretrieve
import os
import json

class prepare_dataset(object):
	def __init__(self, coco):
		self.coco = coco
		self.imgDir = './data/imgs'
		self.annDir = './data/anns'
		self.imgIds = self.coco.getImgIds()
		with open('PythonAPI/cat_dict.json', 'r') as f:
			self.cat_dict = json.load(f)
		self.num_classes = 80

		self.res_dataset = tf.data.Dataset.from_tensor_slices(self.imgIds)
		self.res_dataset = self.res_dataset.map(lambda imgId: tf.py_func(self._parse_res, [imgId], [tf.float32, tf.float32]))
		self.res_dataset = self.res_dataset.map(self._resize_res)
		self.res_dataset = self.res_dataset.batch(16)
		
		self.deep_dataset = tf.data.Dataset.from_tensor_slices(self.imgIds)
		self.deep_dataset = self.deep_dataset.map(lambda imgId: tf.py_func(self._parse_deep, [imgId], [tf.float32, tf.float32]))
		self.deep_dataset = self.deep_dataset.map(self._resize_deep)
		self.deep_dataset = self.deep_dataset.batch(16)

	def _parse_res(self, imgId):
		img = self.coco.loadImgs(imgId)[0]
		fname = os.path.join(self.imgDir, img['file_name'])
		if not os.path.exists(fname):
			urlretrieve(img['coco_url'], fname)
		img_string = tf.read_file(fname)
		img_decoded = tf.image.decode_jpeg(img_string, 3)

		annIds = self.coco.getAnnIds(img['id'])
		anns = self.coco.loadAnns(annIds)
		label = np.zeros(self.num_classes, np.float32)
		for ann in anns:
			label[self.cat_dict['id2c'][ann['category_id']]] = 1

		return img_decoded, label

	def _parse_deep(self, imgId):
		img = self.coco.loadImgs(imgId)[0]
		fname = os.path.join(self.imgDir, img['file_name'])
		if not os.path.exists(fname):
			urlretrieve(img['coco_url'], fname)
		img_string = tf.read_file(fname)
		img_decoded = tf.image.decode_jpeg(img_string, 3)

		annIds = self.coco.getAnnIds(img['id'])
		anns = self.coco.loadAnns(annIds)
		masks = np.zeros([self.num_classes, img.shape[0], img.shape[1]])
		for ann in anns:
			mask = self.coco.annToMask(ann)
			masks[self.cat_dict['id2c'][ann['category_id']]] += mask

		return img_decoded, masks.transpose(1, 2, 0).astype(np.float32)

	def _resize_res(self, img, label):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, [512, 512])
		return img, label

	def _resize_deep(self, img, masks):
		img.set_shape([None, None, 3])
		img = tf.image.resize_images(img, [512, 512])
		masks.set_shape([None, None, self.num_classes])
		masks = tf.image.resize_images(masks, [512, 512])
		return img, masks
