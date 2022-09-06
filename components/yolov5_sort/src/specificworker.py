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
import queue
import time
import traceback
from threading import Thread
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

ROOT = '/home/robocomp/software/Yolov5_StrongSORT_OSNet'  # yolov5 strongsort root directory
WEIGHTS = ROOT + '/weights'
STRONGSORT = ROOT + '/strong_sort'
sys.path.append(str(ROOT))
sys.path.append(str(ROOT + '/yolov5'))  # add yolov5 to PATH
sys.path.append(str(STRONGSORT))  # add strongSORT to PATH
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from utils.augmentations import letterbox
import logging

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1

        yolo_weights = WEIGHTS + '/yolov5m.pt'  # model.pt path(s)
        #strong_sort_weights = WEIGHTS + '/osnet_x0_25_msmt17.pt'  # model.pt path,
        strong_sort_weights = WEIGHTS + '/osnet_x0_25_market1501.pt'
        config_strongsort = STRONGSORT + '/configs/strong_sort.yaml'
        imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        line_thickness = 3  # bounding box thickness (pixels)
        self.half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        self.augment = False  # augmented inference
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.hide_class = False  # hide IDs
        self.display = True

        # Load model
        if eval:
            self.device = torch.device(int(self.device))
        else:
            self.device = select_device(self.device)
        print(yolo_weights)
        self.model = DetectMultiBackend(yolo_weights, device=self.device, dnn=dnn, data=None, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        self.cfg = get_config()
        self.cfg.merge_from_file(config_strongsort)

        self.strongsort = StrongSORT(strong_sort_weights,
                                    self.device,
                                    self.half,
                                    max_dist=self.cfg.STRONGSORT.MAX_DIST,
                                    max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                                    max_age=self.cfg.STRONGSORT.MAX_AGE,
                                    n_init=self.cfg.STRONGSORT.N_INIT,
                                    nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                                    mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                                    ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA)
        
        self.curr_frame = None
        self.prev_frame = None
        self.outputs = None

        # camera read thread
        self.camera_name = 'camera_top'
        self.read_image_queue = queue.Queue(1)
        self.read_thread = Thread(target=self.get_rgb_thread, args=[self.camera_name], name="read_camera_queue", daemon=True)
        self.read_thread.start()

        # Hz
        self.cont = 0
        self.last_time = time.time()
        self.fps = 0

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):
        t1 = time_sync()
        color, im, pred = self.read_image_queue.get()
        t2 = time_sync()
    
        # Process detections
        if pred and pred[0] is not None and len(pred[0]):
            det = pred[0]
            self.curr_frame = color
            annotator = Annotator(color, line_width=2, pil=not ascii)
            #if self.cfg.STRONGSORT.ECC:  # camera motion compensation
            #    self.strongsort.tracker.camera_update(self.prev_frame, self.curr_frame)

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], color.shape).round()

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to strongsort
            t5 = time_sync()
            outputs = self.strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), color)
            t6 = time_sync()

            # draw boxes for visualization
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]
                if self.display:  # Add bbox to image
                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    label = None if self.hide_labels \
                        else (f'{id} {self.model.names[c]}' if self.hide_conf
                              else (f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.model.names[c]} {conf:.2f}'))
                    annotator.box_label(bboxes, label, color=colors(c, True))

            #print(1000.0 * (t2 - t1), 1000.0 * (t6 - t5))
        else:
            self.strongsort.increment_ages()
            LOGGER.info('No detections')

        # Stream results
        color = annotator.result()
        if self.display:
            cv2.imshow('TRACKER', color)
            cv2.waitKey(1)  # 1 millisecond

        self.prev_frame = self.curr_frame
        t7 = time_sync()
        #print(1000.0 * (t7 - t1))

        # FPS
        self.show_fps()

    ############################################################################################
    def get_rgb_thread(self, camera_name: str):
        while True:
            try:
                rgb = self.camerargbdsimple_proxy.getImage(camera_name)
                self.image_captured_time = time.time()
                color = np.frombuffer(rgb.image, dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
                im, _, _ = letterbox(color)  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim                                                                1)
                pred = self.model(im, augment=self.augment, visualize=False)

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                           self.agnostic_nms, max_det=self.max_det)

                if self.cfg.STRONGSORT.ECC:  # camera motion compensation
                   self.strongsort.tracker.camera_update(self.prev_frame, color)

                self.read_image_queue.put([color, im, pred])
            except:
                print("Error communicating with CameraRGBDSimple")
                traceback.print_exc()
                break

    def show_fps(self):
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            print("Freq: ", self.cont, "Hz. Waiting for image")
            self.cont = 0
        else:
            self.cont += 1
    ##########################################################################################
    def startup_check(self):
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompYoloObjects.TBox from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TBox()
        print(f"Testing RoboCompYoloObjects.TKeyPoint from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TKeyPoint()
        print(f"Testing RoboCompYoloObjects.TPerson from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TPerson()
        print(f"Testing RoboCompYoloObjects.TJointData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TJointData()
        print(f"Testing RoboCompYoloObjects.TData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TData()
        QTimer.singleShot(200, QApplication.instance().quit)



    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of getImage method from YoloObjects interface
    #
    def YoloObjects_getImage(self):
        ret = ifaces.RoboCompYoloObjects.RoboCompCameraRGBDSimple.TImage()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of getYoloJointData method from YoloObjects interface
    #
    def YoloObjects_getYoloJointData(self):
        ret = ifaces.RoboCompYoloObjects.TJointData()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of getYoloObjectNames method from YoloObjects interface
    #
    def YoloObjects_getYoloObjectNames(self):
        ret = RoboCompYoloObjects.TObjectNames()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of getYoloObjects method from YoloObjects interface
    #
    def YoloObjects_getYoloObjects(self):
        ret = ifaces.RoboCompYoloObjects.TData()
        #
        # write your CODE here
        #
        return ret
    # ===================================================================
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
    # From the RoboCompYoloObjects you can use this types:
    # RoboCompYoloObjects.TBox
    # RoboCompYoloObjects.TKeyPoint
    # RoboCompYoloObjects.TPerson
    # RoboCompYoloObjects.TJointData
    # RoboCompYoloObjects.TData


