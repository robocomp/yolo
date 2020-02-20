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


class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.sem = True
		self.labels = []
		self.myid = -1
		
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
			self.Period = 50
			self.timer.start(self.Period)
		except:
			traceback.print_exc()
			print("Error reading config params")
			sys.exit()

	@QtCore.Slot()
	def compute(self):
		ret, img = self.cap.read()
		img = cv2.resize(img,(608,608),0 ,0, interpolation = cv2.INTER_LINEAR)   # for full yolo
		#print(img.shape)
		yolo_im = TImage(height=img.shape[0], width=img.shape[1], depth=img.shape[2], image=img)
		#cv2.imshow('Image', img)	

		try:
			objects = self.yoloserver_proxy.processImage(yolo_im)
			self.drawImage(img, objects)	
			#print(len(objects))
			
		except  Exception as e:
			print("error", e)
			
		#try:
			#if self.sem:
				#start = time.time()
				#frame = cv2.resize(frame,(416,416))  #tyne yolo
				
				# fgmask = self.fgbg.apply(frame)
				# kernel = np.ones((5,5),np.uint8)
				# erode = cv2.erode(fgmask, kernel, iterations = 2)
				# dilate = cv2.dilate(erode, kernel, iterations = 2)
				
				#if cv2.countNonZero(dilate) > 0:
				
				# objects = self.processFrame(frame)
				# print(len(objects))
				#self.sem = False
				
				#ms = int((time.time() - start) * 1000)
				#print "elapsed", ms, " ms. FPS: ", int(1000/ms)		except Exception as e:
			#print("error", e)
	
	def processFrame(self, img):
		im = TImage(width=img.shape[1], height=img.shape[0], depth=3, image=img.tostring())
		try:
			myid = self.yoloserver_proxy.processImage(im)
			return myid 
		except  Exception as e:
			print("error 2", e)
	
	def drawImage(self, img, labels):
		if len(labels)>0:
			for box in labels:
				if box.prob > 50:
					p1 = (box.left, box.top)
					p2 = (box.right, box.bot)
					offset = int((p2[1] - p1[1]) / 2)
					pt = (box.left + offset, box.top + offset) 
					cv2.rectangle(img, p1, p2, (0, 0, 255), 4)
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, box.name + " " + str(int(box.prob)) + "%", pt, font, 1, (255, 255, 255), 2)
					
		cv2.imshow('Image', img)
		cv2.waitKey(2)
		
	#
	# newObjects
	#
	# def newObjects(self, id, objs):
	# 	#print "received", len(objs)
	# 	if (id == self.myid):
	# 		self.labels = objs
	# 		#print "#########################################'"
	# 		#print self.labels
	# 		self.sem = True
			

