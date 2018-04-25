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
#from camerareader import CameraReader

class Cap:
	def __init__(self, camera):
		self.stream = requests.get(camera, stream=True)
		
	def read(self):
		bytes = ''
		for chunk in self.stream.iter_content(chunk_size=1024):
			bytes += chunk
			a = bytes.find(b'\xff\xd8')
			b = bytes.find(b'\xff\xd9')
			if a != -1 and b != -1:
				jpg = bytes[a:b+2]
				bytes = bytes[b+2:]
				if len(jpg) > 0:
					img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
					return True, img
				else:
					return False, img


class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		
	def setParams(self, params):
		try:
			camera = params["Camera"]
			if camera == "webcam":
				camera = 0
				self.cap = cv2.VideoCapture(camera)
				print "camera", camera
			else:
				self.cap = Cap(camera)
				
			self.timer.timeout.connect(self.compute)
			self.timer.start(self.Period)
			self.fgbg = cv2.createBackgroundSubtractorMOG2()
		
		except:
			traceback.print_exc()
			print "Error reading config params"
			sys.exit()
			
	

	@QtCore.Slot()
	def compute(self):
		print "---------------------------"
		ret, frame = self.cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		fgmask = self.fgbg.apply(frame)
		kernel = np.ones((5,5),np.uint8)
		erode = cv2.erode(fgmask, kernel, iterations = 2)
		dilate = cv2.dilate(erode, kernel, iterations = 2)
		#k = cv2.waitKey(1)
		
		if cv2.countNonZero(dilate) > 100:
			start = time.time()
			labels = self.processFrame(frame)
			self.drawImage(frame, labels)
			end = time.time()
			print "elapsed", (end - start) * 1000


	def processFrame(self, img):
#		img = cv2.resize(img,(608,608))
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

	@QtCore.Slot(str)
	def slotNewImage(self):
		self.newImage=True
	#	cv2.imshow('i', self.c.img)
	#	if cv2.waitKey(1) == 27:
	#		exit(0)
