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
from camerareader import CameraReader

class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		# cv2.namedWindow("YOLO IA", cv2.WINDOW_AUTOSIZE)

	def setParams(self, params):
		try:
			par = params["CameraURL"]
			print par
			camera_url = par
			print camera_url
			#self.r = requests.get(
			#	'http://158.49.247.240:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream&usr=admin&pwd=opticalflow',
			#	auth=('admin', 'opticalflow'), stream=True)
			# self.r = requests.get(camera_url, auth=('admin', 'opticalflow'), stream=True)
			self.r = requests.get(camera_url,  stream=True)

			if self.r.status_code is not 200:
				print "Chungo"

			self.frameAnt = np.zeros((480, 640), np.uint8)

			self.newImage = False

			self.c = CameraReader(self.r)
			self.c.start()
			self.c.signalNewImage.connect(self.slotNewImage)

			self.timer.timeout.connect(self.compute)
			self.Period = 20
			self.timer.start(self.Period)

		except:
			traceback.print_exc()
			print "Error reading config params"
		return True

	def initVideo(self):
		try:
			# self.vidFile = cv2.VideoCapture("http://158.49.247.184:88/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=admin&pwd=opticalflow")
			# self.vidFile = cv2.VideoCapture("rtsp://admin:opticalflow@158.49.247.240:88/videoMain")
			# self.vidFile = cv2.VideoCapture(0)
			# self.vidFile = cv2.VideoCapture("http://158.49.247.184:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream&usr=admin&pwd=opticalflow")
			# self.vidFile = cv2.VideoCapture("http://158.49.247.240:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream&usr=administrador&pwd=admin")
			# self.ipc = ipCamera('http://158.49.247.240:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream&usr=admin&pwd=opticalflow', user="admin", password="opticalflow")
			pass
		except:
			print "problem opening input stream"
			sys.exit(1)

	@QtCore.Slot()
	def compute(self):
		print "----------------------------------"
		if self.newImage is True:
			start = time.time()
			img = self.c.img
			labels =[]
			thresImg, moving = self.checkMovement(img)
			if moving:
				labels = self.processFrame(img)
				self.drawImage(img, labels)

			self.newImage = False
			end = time.time()
			print "elapsed", (end - start) * 1000

	def checkMovement(self, frame):
		if frame is None:
			return np.zeros((480, 640), np.uint8), False
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		frameDelta = cv2.absdiff(self.frameAnt, gray)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		nonZero = cv2.countNonZero(thresh)
		print nonZero
		self.frameAnt = gray
		return thresh, nonZero > 10

	def processFrame(self, frame):
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#		img = cv2.resize(img,(608,608))
#		img = cv2.resize(img,(320,240))
		im = Image()
		im.w = img.shape[1]
		im.h = img.shape[0]
		#print 'w', im.w, 'h', im.h
		im.data = []
		im.data = img.tostring()
		try:
			# Send image to server
			id = self.yoloserver_proxy.addImage(im)
			#print "id asignado por servidor: ", id

			# Waiting for result+
			while True:
				labels = self.yoloserver_proxy.getData(id)
				if labels.isReady:
					break
				else:
					time.sleep(0.01);
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
