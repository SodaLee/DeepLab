from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import tensorflow as tf
from urllib.request import urlretrieve
import os


dataDir='..'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
imgDir = '../data/imgs'
annDir = '../data/anns'
num_classes = 91

coco=COCO(annFile)

def prepare_imgs(coco, imgIds):
	'''
	return: imgs [batch_size, height, width, channels]
	'''
	imgs = coco.loadImgs(imgIds)
	rimgs = []
	for img in imgs:
		fname = os.path.join(imgDir, img['file_name'])
		try:
			I = io.imread(fname)
		except:
			urlretrieve(img['coco_url'], fname)
			I = io.imread(fname)
		rimgs.append(I)
	return tf.concat(rimgs, -1)


def prepare_labels(coco, imgIds):
	'''
	return: one hot labels [batch_size, num_classes]
	'''
	imgs = coco.loadImgs(imgIds)
	labels = []
	for img in imgs:
		annIds = coco.getAnnIds(img['id'])
		anns = coco.loadAnns(annIds)
		label = np.zeros(num_classes)
		for ann in anns:
			label[ann['category_id']-1] = 1
		labels.append(label)
	return tf.concat(labels, -1)


def prepare_segs(coco, imgIds):
	'''
	return: segmentations [batch_size, height, width, num_classes]
	'''
	imgs = coco.loadImgs(imgIds)
	seg = []
	for img in imgs:
		annIds = coco.getAnnIds(img['id'])
		anns = coco.loadAnns(annIds)
		masks = np.zeros([height, width, num_classes])
		for ann in anns:
			mask = coco.annToMask(ann)
			masks[ann['category_id']-1] += mask
		seg.append(masks)
	return tf.concat(seg, -1)
