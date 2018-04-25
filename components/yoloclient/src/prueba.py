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
import numpy as np
import cv2

cv2.namedWindow("YOLO IA", cv2.WINDOW_AUTOSIZE)

try:
	#	vidFile = cv2.VideoCapture("http://158.49.247.240:88/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=X&pwd=Y&.mjpg")
	vidFile = cv2.VideoCapture("rtsp://admin:opticalflow@158.49.247.240:88/videoMain")
	# vidFile = cv2.VideoCapture("http://158.49.247.240:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream&usr=admin&pwd=opticalflow")
	#vidFile = cv2.VideoCapture(0)
	# vidFile = cv2.VideoCapture("http://158.49.247.184:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream&usr=admin&pwd=opticalflow")
except:
	print "problem opening input stream"
	sys.exit(1)

if not vidFile.isOpened():
	print "capture stream not open"
	sys.exit(1)

while True:
	ret, frame = vidFile.read()  # read first frame, and the return code of the function.
	if ret:
		#self.processFrame(frame)
		cv2.imshow('Image', frame);
		cv2.waitKey(2);
