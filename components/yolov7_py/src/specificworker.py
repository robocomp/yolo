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
import queue

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
from typing import NamedTuple, List, Mapping, Optional, Tuple, Union


#sys.path.append('/home/robocomp/software/ONNX-YOLOv7-Object-Detection')
#from YOLOv7 import YOLOv7

sys.path.append('/home/robocomp/software/TensorRT-For-YOLO-Series')
from utils.utils import preproc, vis
from utils.utils import BaseEngine

from mediapipe.framework.formats import landmark_pb2, detection_pb2, location_data_pb2
import mediapipe as mp

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 5
        if startup_check:
            self.startup_check()
        else:
            # trt
            self.yolo_object_predictor = BaseEngine(engine_path='yolov7-tiny.trt')
            #self.yolo_skeleton_predictor = BaseEngine(engine_path='densenet121_256x256.trt')

            # pose mediapipe
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
            self.mediapipe_human_pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # camera read thread
            self.read_queue = queue.Queue()
            self.read_thread = Thread(target=self.get_rgb_thread, args=["camera_top"], name="read_queue")
            self.read_thread.start()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        rgb = self.read_queue.get()

        t1 = time.time()
        dets = self.yolov7_objects(rgb)
        t2 = time.time()

        objects = self.post_process(dets, rgb)
        #print(len(objects))

        #self.show_data(dets, rgb)
        t3 = time.time()

        #print(1000.0*(t3-t1), 1000.0*(t2-t1), 1000.0*(t3-t2))

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
            except:
                print("Error communicating with CameraRGBDSimple")
                return
            self.read_queue.put(frame)

    def yolov7_objects(self, frame):
        blob, ratio = preproc(frame, (640, 640), None, None)
        data = self.yolo_object_predictor.infer(blob)

        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

        return dets

    def post_process(self, dets, frame):
        data = ifaces.RoboCompYoloObjects.TData()
        data.objects = []
        data.people = []
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
                #self.yolo_object_predictor.class_names[int(final_cls_inds[i])]
                ibox.prob = final_scores[i]
                ibox.left = int(box[0])
                ibox.top = int(box[1])
                ibox.right = int(box[2])
                ibox.bot = int(box[3])
                data.objects.append(ibox)

                # pose
                if final_cls_inds[i] == 0:  # person
                    body_roi = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :].copy()
                    body_roi.flags.writeable = False
                    pose_results = self.mediapipe_human_pose.process(body_roi)
                    roi_rows, roi_cols, _ = body_roi.shape
                    person = ifaces.RoboCompYoloObjects.TPerson()
                    person.joints = {}
                    if pose_results.pose_landmarks:
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            if ((landmark.HasField('visibility') and
                                 landmark.visibility < _VISIBILITY_THRESHOLD) or
                                    (landmark.HasField('presence') and
                                     landmark.presence < _PRESENCE_THRESHOLD)):
                                continue
                            landmark_px = self.normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                           roi_cols, roi_rows,
                                                                           int(box[0]), int(box[1]))
                            if landmark_px:
                                kp = ifaces.RoboCompYoloObjects.TKeyPoint()
                                kp.i = landmark_px[0]
                                kp.j = landmark_px[1]
                                person.joints[idx] = kp
                        data.people.append(person)

        return data

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
            frame = vis(frame, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=self.yolo_object_predictor.class_names)
        cv2.imshow("Detected Objects", frame)
        cv2.waitKey(1)

    def normalized_to_pixel_coordinates(self,
                                        normalized_x: float,
                                        normalized_y: float,
                                        image_width: int,
                                        image_height: int,
                                        roi_offset_x: int,
                                        roi_offset_y: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        # add original image offser
        x_px += roi_offset_x
        y_px += roi_offset_y

        return x_px, y_px
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

    # =============== Methods for Component Implements ==================
    # ===================================================================

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


