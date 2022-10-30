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
from threading import Thread, Event
import traceback
import queue
import json
from typing import NamedTuple, List, Mapping, Optional, Tuple, Union
from shapely.geometry import box
from collections import defaultdict
import itertools

sys.path.append('/home/robocomp/software/TensorRT-For-YOLO-Series')
from utils.utils import preproc, vis
from utils.utils import BaseEngine

sys.path.append('/home/robocomp/software/ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

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

_ROBOLAB_NAMES = ['person', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                  'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                  'toothbrush']

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1
        self.thread_period = 10
        self.display = False

        if startup_check:
            self.startup_check()
        else:
            # trt
            self.yolo_object_predictor = BaseEngine(engine_path='yolov7.trt')

            # byte tracker
            #self.tracker = BYTETracker(frame_rate=30)

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # camera read thread
            self.read_queue = queue.Queue(1)
            self.event = Event()
            # self.read_thread = Thread(target=self.get_rgb_thread, args=["/Shadow/camera_top", self.event],
            #                           name="read_queue", daemon=True)
            self.read_thread = Thread(target=self.get_rgbd_thread, args=["/Shadow/camera_top", self.event],
                                      name="read_queue", daemon=True)
            self.read_thread.start()

            # result data
            self.objects_write = ifaces.RoboCompYoloObjects.TData()
            self.objects_read = ifaces.RoboCompYoloObjects.TData()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            self.display = params["display"] == "true" or params["display"] == "True"
            print("Params read. Starting...", params)
        except:
            print("Error reading config params")
            traceback.print_exc()

        return True

    @QtCore.Slot()
    def compute(self):
        #t1 = time.time()
        rgb, blob, depth, alive_time, period, dfocalx, dfocaly = self.read_queue.get()
        #t2 = time.time()

        dets = self.yolov7_objects(blob)

        if dets is not None:
            boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            #t3 = time.time()

            #tracked_boxes, tracked_scores, tracked_cls_inds, tracked_inds = self.track(boxes, scores, cls_inds)
            #t4 = time.time()

            #self.objects_write = self.post_process(tracked_boxes, tracked_scores, tracked_cls_inds, tracked_inds)
            self.objects_write = self.post_process(boxes, scores, cls_inds, cls_inds, depth, dfocalx, dfocaly, rgb)
            #t5 = time.time()

            self.objects_write, self.objects_read = self.objects_read, self.objects_write

            if self.display:
#                rgb = self.display_data(rgb, tracked_boxes, tracked_scores, tracked_cls_inds, tracked_inds,
#                                        class_names=self.yolo_object_predictor.class_names)
                rgb = self.display_data(rgb, boxes, scores, cls_inds, cls_inds,
                                        class_names=self.yolo_object_predictor.class_names)

        if self.display:
            cv2.imshow("Detected Objects", rgb)
            cv2.waitKey(1)

        #print(1000.0*(t2-t1), 1000.0*(t3-t2), 1000.0*(t4-t3), 1000.0*(t5-t4))

        # FPS
        try:
            self.show_fps(alive_time, period)
        except KeyboardInterrupt:
            self.event.set()


    ###########################################################################################3

    def get_rgb(self, name):
        try:
            rgb = self.camerargbdsimple_proxy.getImage(name)
            frame = np.frombuffer(rgb.image, dtype=np.uint8)
            frame = frame.reshape((rgb.height, rgb.width, 3))
        except:
            print("Error communicating with CameraRGBDSimple")
            traceback.print_exc()
            return
        return frame

    def get_rgb_thread(self, camera_name: str, event: Event):
        while not event.isSet():
            try:
                rgb = self.camerargbdsimple_proxy.getImage(camera_name)
                frame = np.frombuffer(rgb.image, dtype=np.uint8)
                frame = frame.reshape((rgb.height, rgb.width, 3))
                blob = self.pre_process(frame, (640, 640))
                delta = int(1000 * time.time() - rgb.alivetime)
                self.read_queue.put([frame, blob, delta, rgb.period])
                event.wait(self.thread_period/1000)
            except:
                print("Error communicating with CameraRGBDSimple")
                traceback.print_exc()

    def get_rgbd_thread(self, camera_name: str, event: Event):
        while not event.isSet():
            try:
                rgbd = self.camerargbdsimple_proxy.getAll(camera_name)
                rgb_frame = np.frombuffer(rgbd.image.image, dtype=np.uint8).reshape((rgbd.image.height, rgbd.image.width, 3))
                blob = self.pre_process(rgb_frame, (640, 640))  #TODO: change to vars
                delta = int(1000 * time.time() - rgbd.image.alivetime)  #TODO: one alivetime and period per frame
                depth_frame = np.frombuffer(rgbd.depth.depth, dtype=np.float32).reshape((rgbd.depth.height, rgbd.depth.width, 1))
                self.read_queue.put([rgb_frame, blob, depth_frame, delta, rgbd.image.period, rgbd.depth.focalx, rgbd.depth.focaly])
                event.wait(self.thread_period/1000)
            except:
                print("Error communicating with CameraRGBDSimple")
                traceback.print_exc()

    ###############################################################
    def pre_process(self, image, input_size, swap=(2, 0, 1)):
        padded_img = np.ones((input_size[0], input_size[1], 3))
        img = np.array(image).astype(np.float32)
        padded_img[: int(img.shape[0]), : int(img.shape[1])] = img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def yolov7_objects(self, blob):
        data = self.yolo_object_predictor.infer(blob)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        return dets

    def track(self, boxes, scores, cls_inds):
        final_boxes = []
        final_scores = []
        final_cls_ids = []
        final_ids = []
        people_inds = [i for i, cls in enumerate(cls_inds) if cls == 0]   # index of elements with value 0
        people_scores = scores[people_inds]
        people_boxes = boxes[people_inds]
        online_targets = self.tracker.update2(people_scores, people_boxes, [640, 640], (640, 640))
        tracked_boxes = []
        tracked_scores = []
        tracked_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 10 and not vertical:
                tracked_boxes.append(tlwh)
                tracked_ids.append(tid)
                tracked_scores.append(t.score)

        if tracked_boxes:
            tracked_boxes = np.asarray(tracked_boxes)
            tracked_boxes[:, 2] = tracked_boxes[:, 0] + tracked_boxes[:, 2]
            tracked_boxes[:, 3] = tracked_boxes[:, 1] + tracked_boxes[:, 3]

            # we replace the original person boxes by the tracked ones
            non_people_cls_inds = [i for i, cls in enumerate(cls_inds) if cls != 0]  # index of non-person elements
            final_boxes = np.append(boxes[non_people_cls_inds], tracked_boxes, axis=0)  # non-person boxes + tracked people
            final_scores = np.append(scores[non_people_cls_inds], tracked_scores, axis=0)
            final_ids = np.append(np.full(len(non_people_cls_inds), -1), tracked_ids, axis=0)
            final_cls_ids = np.append(cls_inds[non_people_cls_inds], np.zeros(len(people_inds)), axis=0)
        return final_boxes, final_scores, final_cls_ids, final_ids

    def post_process(self, final_boxes, final_scores, final_cls_inds, final_inds, depth, focalx, focaly, rgb):   # copy to interface
        data = ifaces.RoboCompYoloObjects.TData()
        data.objects = []
        data.people = []
        depth_to_rgb_factor_rows = rgb.shape[0] // depth.shape[0]
        depth_to_rgb_factor_cols = rgb.shape[1] // depth.shape[1]

        for i in range(len(final_boxes)):
            box = final_boxes[i]
            # if score < conf:
            #    continue
            ibox = ifaces.RoboCompYoloObjects.TBox()
            ibox.type = int(final_cls_inds[i])
            ibox.id = int(final_inds[i])
            ibox.score = final_scores[i]
            ibox.left = int(box[0])
            ibox.top = int(box[1])
            ibox.right = int(box[2])
            ibox.bot = int(box[3])
            #compute x,y,z coordinates in camera CS of bbox's center
            left = ibox.left // depth_to_rgb_factor_cols
            right = ibox.right // depth_to_rgb_factor_cols
            top = ibox.top // depth_to_rgb_factor_rows
            bot = ibox.bot // depth_to_rgb_factor_rows
            roi = depth[top: top+bot, left: left+right]
            cx_roi = int(roi.shape[1]/2)
            cy_roi = int(roi.shape[0]/2)
            #ibox.depth = float(np.median(roi[cy_roi-20:cy_roi+20, cx_roi-10:cx_roi+10]))*1000
            ibox.depth = float(np.median(depth[top: top+bot, left: left+right])) * 1000
            #ibox.depth = float(depth[cy_roi, cx_roi])*1000
            #ibox.depth = float(np.min(roi)) * 1000

            cx_i = (ibox.left + ibox.right)/2
            cy_i = (ibox.top + ibox.bot)/2
            cx = cx_i - depth.shape[1]/2
            cy = cy_i - depth.shape[0]/2
            # if depth plane gives length of optical ray then
            x = cx * ibox.depth / np.sqrt(cx*cx + focalx*focalx)
            z = cy * ibox.depth / np.sqrt(cy*cy + focaly*focaly)  # Z upwards
            proy = np.sqrt(ibox.depth*ibox.depth-z*z)
            y = np.sqrt(x*x+proy*proy)

            # if deph plane in RGBD gives Y coordinate then
            #y = ibox.depth
            #x = cx * ibox.depth / focalx
            #z = cy * ibox.depth / focaly  # Z upwards
            ibox.x = x
            ibox.y = y
            ibox.z = z
            #print(int(ibox.depth), ibox.type, int(ibox.x), int(ibox.y), int(ibox.z), cx)
            data.objects.append(ibox)

        #data.objects = self.nms(data.objects)
        return data

    def nms(self, objects):
        d = defaultdict(list)
        for obj in objects:
            d[obj.type].append(obj)
        removed = []
        for typ, same_type_objs in d.items():
            power_set = itertools.combinations(same_type_objs, 2)  # possible combs of same type
            # compute IOU
            for a, b in power_set:
                p1 = box(a.left, a.top, a.right, a.bot)  # shapely object
                p2 = box(b.left, b.top, b.right, b.bot)
                intersect = p1.intersection(p2).area / p1.union(p2).area
                if intersect > 0.65:
                    removed.append(a.id if abs(a.prob) > abs(b.prob) else b.id)
        ret = []
        for obj in objects:
            if obj.id not in removed:
                ret.append(obj)
        return ret

    def show_fps(self, alive_time, period):
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000./self.cont)
            delta = (-1 if (period - cur_period) < -1 else (1 if (period - cur_period) > 1 else 0))
            print("Freq:", self.cont, "Hz. Alive_time:", alive_time, "ms. Img period:", int(period),
                  "ms. Curr period:", cur_period, "ms. Inc:", delta, "Timer:", self.thread_period)
            self.thread_period = np.clip(self.thread_period+delta, 0, 200)
            self.cont = 0
        else:
            self.cont += 1

    def display_data(self, img, boxes, scores, cls_inds, inds, class_names=None):
        #print(len(inds), len(boxes))
        for i in range(len(boxes)):
            if inds[i] == -1:
                continue
            bb = boxes[i]
            cls_ids = int(cls_inds[i])
            ids = int(inds[i])
            score = scores[i]
            x0 = int(bb[0])
            y0 = int(bb[1])
            x1 = int(bb[2])
            y1 = int(bb[3])
            color = (_COLORS[ids] * 255).astype(np.uint8).tolist()
            text = 'Class: {} - Score: {:.1f}% - ID: {}'.format(class_names[cls_ids], score*100, ids)
            txt_color = (0, 0, 0) if np.mean(_COLORS[ids]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[ids] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

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
        return self.objects_read

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
