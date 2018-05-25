from PythonAPI.prepare_data import prepare_dataset
import deeplab
from PythonAPI.pycocotools.coco import COCO

dataDir = '.'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)
Datasets = prepare_dataset(coco)
res_iter = Datasets.res_dataset.make_initializable_iterator()
deep_iter = Datasets.deep_dataset.make_initializable_iterator()

_imgs = tf.placeholder([batch_size, None, None, 3], tf.float32)
_labels = tf.placeholder([batch_size, num_classes], tf.float32)
_gt = tf.placeholder([batch_size, None, None, num_classes], tf.float32)
_deeplab = deeplab.deeplab_v3_plus(_imgs, [128, 64], [48, num_classes], num_classes)
res_out = _deeplab.get_dense()
res_loss =
res_op =

pred_out = _deeplab.get_pred()
pred_loss =
pred_op =

def train(train_type='Resnet'):
	with tf.Session() as sess:
		if train_type == 'Resnet':
		elif train_type == 'Deeplab':
		else:
			pass

if __name__ == '__main__':
	train('Resnet')
