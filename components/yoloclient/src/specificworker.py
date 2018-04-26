#
# Copyright (C) 2017 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

import sys, os, traceback, time
from PySide import *
from genericworker import *
import numpy as np
import cv2
import requests
import Queue, threading
#from camerareader import CameraReader

class Cap(threading.Thread):
	def __init__(self, camera, myqueue):
		super(Cap,self).__init__()
		self.stream = requests.get(camera, stream=True)
		self.myqueue = myqueue
		if self.stream.status_code is not 200:
			print "Error connecting to stream ", camera
			sys.exit(1)
		
	def run(self):
		byte = bytes()
		for chunk in self.stream.iter_content(chunk_size=1024):
			byte += chunk
			a = byte.find(b'\xff\xd8')
			b = byte.find(b'\xff\xd9')
			if a != -1 and b != -1:
				jpg = byte[a:b+2]
				byte = byte[b+2:]
				if len(jpg) > 0:
					img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
					self.myqueue.put(img)	


class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.myqueue = Queue.Queue()
		
	def setParams(self, params):
		try:
			camera = params["Camera"]
			if camera == "webcam":
				camera = 0
				self.cap = cv2.VideoCapture(camera)
			else:
				self.cap = Cap(camera, self.myqueue)
				self.cap.start()
				
			self.fgbg = cv2.createBackgroundSubtractorMOG2()
			self.timer.timeout.connect(self.compute)
			self.Period = 5
			self.timer.start(self.Period)
		
		except:
			traceback.print_exc()
			print "Error reading config params"
			sys.exit()
			
	@QtCore.Slot()
	def compute(self):
		start = time.time()
		frame = cv2.cvtColor(self.myqueue.get(), cv2.COLOR_BGR2RGB)
		fgmask = self.fgbg.apply(frame)
		kernel = np.ones((5,5),np.uint8)
		erode = cv2.erode(fgmask, kernel, iterations = 2)
		dilate = cv2.dilate(erode, kernel, iterations = 2)
		
		if cv2.countNonZero(dilate) > 100:
			labels = self.processFrame(frame)
			self.drawImage(frame, labels)
		ms = int((time.time() - start) * 1000)
		print "elapsed", ms, " ms. FPS: ", int(1000/ms)

	def processFrame(self, img):
		img = cv2.resize(img,(608,608))
		im = Image()
		im.w = img.shape[1]
		im.h = img.shape[0]
		im.data = img.tostring()
		try:
			# Send image to server
			id = self.yoloserver_proxy.addImage(im)
			# Waiting for result+
			while True:
				labels = self.yoloserver_proxy.getData(id)
				if labels.isReady:
					break
				else:
					time.sleep(0.001);					
			return labels

		except  Exception as e:
			print "error", e

	def drawImage(self, img, labels):
		if labels:
			for box in labels.lBox:
				if box.prob > 35:
					p1 = (int(box.x), int(box.y))
					p2 = (int(box.w), int(box.h))
					pt = (int(box.x), int(box.y) + (p2[1] - p1[1]) / 2)
					cv2.rectangle(img, p1, p2, (0, 0, 255), 4)
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, box.label + " " + str(int(box.prob)) + "%", pt, font, 1, (255, 255, 255), 2)
		cv2.imshow('Image', img);
		cv2.waitKey(2);
