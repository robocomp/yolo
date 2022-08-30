#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 by YOUR NAME HERE
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
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import time
import math
import cv2
from threading import Thread
import queue
import json
from typing import NamedTuple, List, Mapping, Optional, Tuple, Union

sys.path.append('/home/robocomp/software/TensorRT-For-YOLO-Series')
from utils.utils import preproc, vis
from utils.utils import BaseEngine

# from mediapipe.framework.formats import landmark_pb2, detection_pb2, location_data_pb2
# import mediapipe as mp

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

_OBJECT_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep',
                 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 5
        if startup_check:
            self.startup_check()
        else:
            # trt
            self.yolo_object_predictor = BaseEngine(engine_path='yolov7-tiny.trt')

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # camera read thread
            self.read_queue = queue.Queue(1)
            self.read_thread = Thread(target=self.get_rgb_thread, args=["camera_top"], name="read_queue")
            self.read_thread.start()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

            # result data
            self.write_queue = queue.Queue(1)
            self.data = ifaces.RoboCompYoloObjects.TData()

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        rgb = self.read_queue.get()

        # t1 = time.time()
        dets = self.yolov7_objects(rgb)
        # t2 = time.time()

        objects = self.post_process(dets, rgb)
        # print(len(objects))

        self.show_data(dets, rgb)
        # print(len(objects.objects), len(objects.people))
        # t3 = time.time()

        try:
            self.write_queue.put_nowait(objects)
        except:
            pass
        # print(1000.0*(t3-t1), 1000.0*(t2-t1), 1000.0*(t3-t2))

        # FPS
        self.show_fps()

        return True

    ###########################################################################################3

    def get_rgb(self, name):
        try:
            rgb = self.camerargbdsimple_proxy.getImage(name)
            frame = np.frombuffer(rgb.image, dtype=np.uint8)
            frame = frame.reshape((rgb.height, rgb.width, 3))
        except:
            print("Error communicating with CameraRGBDSimple")
            return
        return frame

    def get_rgb_thread(self, camera_name: str):
        while True:
            try:
                rgb = self.camerargbdsimple_proxy.getImage(camera_name)
                frame = np.frombuffer(rgb.image, dtype=np.uint8)
                frame = frame.reshape((rgb.height, rgb.width, 3))
                self.read_queue.put(frame)
            except:
                print("Error communicating with CameraRGBDSimple")
                return

    ###############################################################
    def yolov7_objects(self, frame):
        blob, ratio = preproc(frame, (640, 640), None, None)
        data = self.yolo_object_predictor.infer(blob)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        return dets

    def post_process(self, dets, frame):
        self.data.objects = []
        self.data.people = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                # if score < conf:
                #    continue

                # copy to interface
                ibox = ifaces.RoboCompYoloObjects.TBox()
                ibox.type = int(final_cls_inds[i])
                ibox.id = i
                ibox.prob = final_scores[i]
                ibox.left = int(box[0])
                ibox.top = int(box[1])
                ibox.right = int(box[2])
                ibox.bot = int(box[3])
                self.data.objects.append(ibox)
        return self.data

    def show_fps(self):
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            print("Freq: ", self.cont, "Hz. Waiting for image")
            self.cont = 0
        else:
            self.cont += 1

    def show_data(self, dets, frame):
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            frame = vis(frame, final_boxes, final_scores, final_cls_inds, conf=0.5,
                        class_names=self.yolo_object_predictor.class_names)
        #
        cv2.imshow("Detected Objects", frame)
        cv2.waitKey(1)

    ############################################################################################
    def startup_check(self):
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompHumanCameraBody.TImage from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.TImage()
        print(f"Testing RoboCompHumanCameraBody.TGroundTruth from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.TGroundTruth()
        print(f"Testing RoboCompHumanCameraBody.KeyPoint from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.KeyPoint()
        print(f"Testing RoboCompHumanCameraBody.Person from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.Person()
        print(f"Testing RoboCompHumanCameraBody.PeopleData from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.PeopleData()
        QTimer.singleShot(200, QApplication.instance().quit)

    # ===================================================================
    #
    # IMPLEMENTATION of getImage method from YoloObjects interface
    #
    def YoloObjects_getImage(self):
        # ret = RoboCompYoloObjects.RoboCompCameraRGBDSimple::TImage()
        #
        # write your CODE here
        #
        return ret
        #

    # IMPLEMENTATION of getYoloJointNames method from YoloObjects interfa
    #
    def YoloObjects_getYoloJointData(self):
        ret = ifaces.RoboCompYoloObjects.TJointData()
        ret.jointNames = {}
        for i, jnt in enumerate(_JOINT_NAMES):
            ret.jointNames[i] = jnt
        ret.connections = []
        for a, b in _CONNECTIONS:
            conn = ifaces.RoboCompYoloObjects.TConnection()
            conn.first = a
            conn.second = b
            ret.connections.append(conn)
        return ret

    # IMPLEMENTATION of getYoloObjectNames method from YoloObjects interf
    #
    def YoloObjects_getYoloObjectNames(self):

        return self.yolo_object_predictor.class_names

    # IMPLEMENTATION of getYoloObjects method from YoloObjects interface
    #
    def YoloObjects_getYoloObjects(self):
        return self.write_queue.get()

    ######################
    # From the RoboCompCameraRGBDSimple you can call this methods:
    # self.camerargbdsimple_proxy.getAll(...)
    # self.camerargbdsimple_proxy.getDepth(...)
    # self.camerargbdsimple_proxy.getImage(...)

    ######################
    # From the RoboCompCameraRGBDSimple you can use this types:
    # RoboCompCameraRGBDSimple.TImage
    # RoboCompCameraRGBDSimple.TDepth
    # RoboCompCameraRGBDSimple.TRGBD

    ######################
    # From the RoboCompHumanCameraBody you can use this types:
    # RoboCompHumanCameraBody.TImage
    # RoboCompHumanCameraBody.TGroundTruth
    # RoboCompHumanCameraBody.KeyPoint
    # RoboCompHumanCameraBody.Person
    # RoboCompHumanCameraBody.PeopleData

# MEDIAPIPE DATA   https://google.github.io/mediapipe/solutions/pose.html
# WHITE_COLOR = (224, 224, 224)
# BLACK_COLOR = (0, 0, 0)
# RED_COLOR = (0, 0, 255)
# GREEN_COLOR = (0, 128, 0)
# BLUE_COLOR = (255, 0, 0)
# _PRESENCE_THRESHOLD = 0.5
# _VISIBILITY_THRESHOLD = 0.5
# _JOINT_NAMES = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
#                 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
#                 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
#                 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee',
#                 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
#                 'right_foot_index']
# _CONNECTIONS = [[1, 2], [2, 3], [3, 7], [0, 1], [0, 4], [4, 5], [5, 6], [6, 8], [10, 9], [12, 14], [14, 16],
#                 [16, 22], [16, 20], [16, 18], [18, 20], [11, 13], [13, 15], [15, 21], [15, 19], [15, 17], [19, 17],
#                 [12, 24], [24, 26], [26, 28], [28, 32], [32, 30], [28, 30], [11, 23], [24, 23], [23, 25], [25, 27],
#                 [27, 29], [29, 31], [27, 31]]