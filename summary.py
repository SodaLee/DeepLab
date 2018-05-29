import csv
import os

class summarizer(object):
	def __init__(self, path, headers, steps, restore = True, verbose = True):
		dirname, filename = os.path.split(path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		self.f = open(path, 'a+' if restore else 'w+')
		self.writer = csv.DictWriter(self.f, headers)
		self.headers = headers
		self.steps = steps
		if not restore:
			self.writer.writeheader()
		self.buffer = []
		self.verbose = verbose

	def __del__(self):
		self.flush()
		self.f.close()
	
	def flush(self):
		self._endline()
		self.writer.writerows(self.buffer)
		self.buffer = []

	def _newline(self):
		self.buffer.append(dict.fromkeys(self.headers, [0 for i in self.headers]))
		self.cnt = dict.fromkeys(self.headers, [0 for i in self.headers])
		self.step = 0

	def _endline(self):
		if len(self.buffer) == 0:
			return
		for key in self.buffer[-1]:
			if self.cnt[key] > 0:
				self.buffer[-1][key] /= self.cnt[key]

	def summary(self, **kwargs):
		if len(self.buffer) == 0:
			self._newline()

		for key, val in kwargs.items():
			self.buffer[-1][key] += val
			self.cnt[key] += 1
		self.step += 1

		if self.step == self.steps:
			self._endline()
			if self.verbose:
				print(self.buffer[-1])
			self._newline()
		
