#
# Copyright (C) 2018 by YOUR NAME HERE
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

from PySide import QtGui, QtCore
from genericworker import *
import numpy as np
import cv2


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.sem = True
		self.labels = []
		self.myid = -1
		self.Period = 20
		self.timer.start(self.Period)

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
		ret, frame = self.cap.read();
		frame = cv2.resize(frame,(608,608))   # for full yolo
		self.drawImage(frame, self.labels)	
		
		try:
			if self.sem:
				start = time.time()
				#frame = cv2.resize(frame,(416,416))  #tyne yolo
				fgmask = self.fgbg.apply(frame)
				kernel = np.ones((5,5),np.uint8)
				erode = cv2.erode(fgmask, kernel, iterations = 2)
				dilate = cv2.dilate(erode, kernel, iterations = 2)
				
				#if cv2.countNonZero(dilate) > 0:
				
				self.myid = self.processFrame(frame)
				self.sem = False
				
				#ms = int((time.time() - start) * 1000)
				#print "elapsed", ms, " ms. FPS: ", int(1000/ms)
		except Exception as e:
			print "error", e
	
	def processFrame(self, img):
		im = TImage()
		im.width = img.shape[1]
		im.height = img.shape[0]
		im.depth = 3
		im.image = img.tostring()
		try:
			myid = self.yoloserver_proxy.processImage(im)
			return myid 
		except  Exception as e:
			print "error", e
	
	def drawImage(self, img, labels):
		if len(labels)>0:
			for box in labels:
				if box.prob > 50:
					p1 = (int(box.left), int(box.top))
					p2 = (int(box.right), int(box.bot))
					pt = (int(box.left), int(box.top) + (p2[1] - p1[1]) / 2)
					cv2.rectangle(img, p1, p2, (0, 0, 255), 4)
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, box.name + " " + str(int(box.prob)) + "%", pt, font, 1, (255, 255, 255), 2)
		cv2.imshow('Image', img);
		cv2.waitKey(2);
	
	#
	# newObjects
	#
	def newObjects(self, id, objs):
		#print "received", len(objs)
		if (id == self.myid):
			self.labels = objs
			#print "#########################################'"
			#print self.labels
			self.sem = True
			

