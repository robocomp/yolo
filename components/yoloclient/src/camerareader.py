import sys, os, traceback, time
import numpy as np
import cv2
from PySide.QtCore import *

class CameraReader(QThread):
	signalNewImage = Signal()
	def __init__(self, stream):
		super(CameraReader, self).__init__()
		self.r = stream
		print "init", self.r

	def run(self):
		print "gola"
		byte = bytes()
		for chunk in self.r.iter_content(chunk_size=1024):
			byte += chunk
			a = byte.find(b'\xff\xd8')
			b = byte.find(b'\xff\xd9')
			if a != -1 and b != -1:
				jpg = byte[a:b + 2]
				byte = byte[b + 2:]
				self.img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
				self.signalNewImage.emit()
		else:
			print("Received unexpected status code {}".format(r.status_code))