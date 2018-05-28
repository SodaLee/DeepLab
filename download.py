import os
from urllib.request import urlretrieve

def cbk(a,b,c):
	per = 100.0*a*b/c
	if per > 100:
		per = 100
	print('downloaded: %.3f%%' % per)
url = 'http://images.cocodataset.org/zips/train2017.zip'
wdir = './train2017.zip'
urlretrieve(url, wdir, cbk)