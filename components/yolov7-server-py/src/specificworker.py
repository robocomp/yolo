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
#    (at your self.option) any later version.
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
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from typing import NamedTuple

sys.path.append('/home/robocomp/software/yolov7')
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier
from utils.general import scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


class Options(NamedTuple):
    weights: str = 'yolov7-tiny.pt'
    #weights: str = 'yolov7-tiny.pt'
    source: str = '0'
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = '0'
    view_img: bool = True
    classes: int = 0
    agnostic_nms: bool = True
    augment: bool = True
    update: bool = True


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50
        if startup_check:
            self.startup_check()
        else:

            self.opt = Options()
            self.init_detect()
            print("Init_detect completed")

            self.ext_image = ifaces.RoboCompYoloServer.TImage()
            self.new_ext_image = False
            self.objects = []
            # Hz
            self.cont = 0
            self.last_time = time.time()

            self.display = False

            self.timer.timeout.connect(self.compute)
            #self.timer.setSingleShot(True)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        self.display = params["display"] == "true" or params["display"] == "True"
        return True

    @QtCore.Slot()
    def compute(self):
        self.detect()
        return True

#######################################################################################################
    def init_detect(self, save_img=False):
        source, weights, self.view_img, self.imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.img_size
        webcam = source.isnumeric()

        # Initialize
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = True
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        #if webcam:
        #    view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

    def detect(self):
        path = '0'
        # get image img0
        if self.new_ext_image:
            t0 = time.time()
            color = np.frombuffer(self.ext_image.image, dtype=np.uint8)
            color = color.reshape((self.ext_image.height, self.ext_image.width, 3))
            img, ratio, (dw, dh) = self.letterbox(color)

            # Convert img format
            img = np.stack([img], 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            # if self.device.type != 'cpu' and (
            #         old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            #     old_img_b = img.shape[0]
            #     old_img_h = img.shape[2]
            #     old_img_w = img.shape[3]
            #     for i in range(3):
            #         model(img, augment=self.opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,  #classes=self.opt.classes,#
                                       agnostic=self.opt.agnostic_nms)
            t3 = time_synchronized()

            # Second-state classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, color)

            # Process detections
            self.objects = []
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path[i], '%g: ' % i, color.copy()

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.display:  # Add bbox to image
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                        box = ifaces.RoboCompYoloServer.Box()
                        box.name = self.names[int(cls)]
                        box.prob = float(conf)
                        box.left = int(xyxy[0])
                        box.top = int(xyxy[1])
                        box.right = int(xyxy[2])
                        box.bot = int(xyxy[3])
                        self.objects.append(box)

                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}s) NMS')

                # Stream results
                if self.display:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

            self.new_ext_image = False
            #print(f'Done. ({time.time() - t0:.3f}s)')

        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            print("Freq: ", self.cont, "Hz. Waiting for image")
            self.cont = 0
        else:
            self.cont += 1


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

       # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    #################################################################################33
    def startup_check(self):
        print(f"Testing RoboCompYoloServer.TImage from ifaces.RoboCompYoloServer")
        test = ifaces.RoboCompYoloServer.TImage()
        print(f"Testing RoboCompYoloServer.Box from ifaces.RoboCompYoloServer")
        test = ifaces.RoboCompYoloServer.Box()
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Interface methods ==================
    # ====================================================
    #
    # IMPLEMENTATION of processImage method from YoloServer interface
    #
    def YoloServer_processImage(self, img):

        ret = ifaces.RoboCompYoloServer.Objects()
        if not self.new_ext_image:
            self.ext_image = img
            # wait for detection
            while not self.ext_image:  # tags cannot be computed in this thread
                pass
            ret = self.objects
            self.new_ext_image = True

        return ret
    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompYoloServer you can use this types:
    # RoboCompYoloServer.TImage
    # RoboCompYoloServer.Box


