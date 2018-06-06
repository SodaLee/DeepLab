import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import os
import numpy as np
import math

def draw_curve(path, figpath, keys = None, xlabel = 'x', ylabel = 'y', title = None, legend_loc = 'best'):
	assert os.path.exists(path), "no such file %s"%path
	if not os.path.exists(os.path.split(figpath)[0]):
		os.makedirs(os.path.split(figpath)[0])
	x = []
	if keys is not None:
		y = [[] for i in keys]
	with open(path, "r") as f:
		reader = csv.DictReader(f)
		for i, row in enumerate(reader):
			x.append(i)
			if keys is None:
				keys = [key for key in row]
				y = [[] for k in keys]
			for j, key in enumerate(keys):
				y[j].append(float(row[key]))
	
	fig = plt.figure()
	axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	if title is not None:
		axes.set_title(title)
	for i, key in enumerate(keys):
		axes.plot(x, y[i], label = key)
	axes.legend(loc = legend_loc)
	plt.draw()
	plt.savefig(figpath)
	plt.close("all")

def draw_raw_image(raw_image, figpath):
	fig = plt.figure(dpi = 160)
	axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
	axes.imshow(raw_image)
	plt.draw()
	plt.savefig(figpath)
	plt.close("all")

def draw_image(img, figpath, keys = None):
	assert len(img.shape) == 3, "unsupported image shape %s"%(img.shape)
	if not os.path.exists(os.path.split(figpath)[0]):
		os.makedirs(os.path.split(figpath)[0])
	nclass = img.shape[-1]
	cuber = int(math.ceil(math.pow(nclass, 1.0 / 3)))
	sqrcuber = cuber * cuber
	colors = np.empty([nclass, 3], dtype = np.float32)
	stride1 = 1. / ((nclass - 1) // sqrcuber)
	stride2 = 1. / ((sqrcuber - 1) // cuber)
	stride3 = 1. / (cuber - 1)
	for i in range(nclass):
		r = i // sqrcuber
		g = (i % sqrcuber) // cuber
		b = (i % sqrcuber) % cuber
		colors[i] = np.array([r * stride1, g * stride2, b * stride3], dtype = np.float32)
	colors = 1. - colors
	colors[-1] = [0, 0, 0]

	img_hotkey = np.reshape(np.argmax(img, axis = -1), [-1])
	img_onehot = np.reshape(np.eye(nclass)[img_hotkey], img.shape)
	res = np.matmul(img_onehot, colors)

	fig = plt.figure(dpi = 160)
	if img.shape[0] > img.shape[1]:
		h = 0.9 - nclass // 20 * 0.1
		w = h / img.shape[0] * img.shape[1]
		axes = fig.add_axes([0.1, 0.1, w, h])
	else:
		w = 0.9 - (nclass + 7) // 8 * 0.033
		h = w / img.shape[1] * img.shape[0]
		axes = fig.add_axes([0.1, 0.9 - h, w, h])
	axes.imshow(res)
	if keys is not None:
		patches = []
		for i in range(nclass):
			patches.append(mpatches.Patch(color=colors[i], label=keys[i]))
		if img.shape[0] > img.shape[1]:
			axes.legend(handles = patches, loc = "center right", bbox_to_anchor=(1.2, 0.5), ncol = nclass // 20)
		else:
			axes.legend(handles = patches, loc = "lower center", bbox_to_anchor=(0.5, -0.4), ncol = 8)
	plt.draw()
	plt.savefig(figpath)
	plt.close("all")
