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

def draw_image(img, figpath, keys = None, legend_loc = 'lower center'):
	assert len(img.shape) == 3, "unsupported image shape %s"%(img.shape)
	if not os.path.exists(os.path.split(figpath)[0]):
		os.makedirs(os.path.split(figpath)[0])
	nclass = img.shape[-1]
	cuber = int(math.ceil(math.pow(nclass, 1.0 / 3)))
	sqrcuber = cuber * cuber
	colors = np.empty([nclass, 3], dtype = np.float32)
	for i in range(nclass):
		r = int(i / sqrcuber)
		g = int((i % sqrcuber) / cuber)
		b = (i % sqrcuber) % cuber
		colors[i] = cuber - 1 - np.array([r, g, b], dtype = np.float32)
	colors /= np.amax(colors)

	img_hotkey = np.reshape(np.argmax(img, axis = -1), [-1])
	img_onehot = np.reshape(np.eye(nclass)[img_hotkey], img.shape)
	res = np.matmul(img_onehot, colors)

	fig = plt.figure()
	axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
	axes.imshow(res)
	if keys is not None:
		patches = []
		for i in range(nclass):
			patches.append(mpatches.Patch(color=colors[i], label=keys[i]))
		axes.legend(handles=patches, loc = legend_loc)
	plt.draw()
	plt.savefig(figpath)
	plt.close("all")
