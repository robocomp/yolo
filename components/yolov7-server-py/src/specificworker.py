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
import time, math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from typing import NamedTuple, List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import landmark_pb2, detection_pb2, location_data_pb2
import dataclasses

sys.path.append('/home/robocomp/software/yolov7')
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier
from utils.general import scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import mediapipe as mp

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

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

@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2

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

            # initialize  estimators
            # pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_pose = mp.solutions.pose
            self.mediapipe_human_pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            # face
            self.mp_face = mp.solutions.face_detection
            self.mediapipe_face = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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
        #self.media_pipe()
        return True

#######################################################################################################
    def media_pipe(self):
        if self.new_ext_image:
            t0 = time.time()
            color = np.frombuffer(self.ext_image.image, dtype=np.uint8)
            color = color.reshape((self.ext_image.height, self.ext_image.width, 3))
            color.flags.writeable = False
            image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = self.mp_objectron.process(image)

            # Draw the box landmarks on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    self.mp_drawing.draw_landmarks(
                        image, detected_object.landmarks_2d, self.mp_objectron.BOX_CONNECTIONS)
                    self.mp_drawing.draw_axis(image, detected_object.rotation,
                                         detected_object.translation)
            cv2.imshow("Objectron", image)
            cv2.waitKey(1)
            self.new_ext_image = False


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
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        # Whatever
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

    def detect(self):
        path = '0'
        # get image img0
        if self.new_ext_image:
            t0 = time.time()
            color = np.frombuffer(self.ext_image.image, dtype=np.uint8)
            color = color.reshape((self.ext_image.height, self.ext_image.width, 3))
            #depth = np.frombuffer(self.ext_image.depth, dtype=np.float32)
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

            # Process detections
            self.objects = []   # use a double buffer to avoid deleting before read in interface
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path[i], '%g: ' % i, color.copy()

                #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if cls == 0:  # if person extract skeleton
                            body_roi = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                            roi_rows, roi_cols, _ = body_roi.shape
                            body_roi.flags.writeable = False
                            body_roi = cv2.cvtColor(body_roi, cv2.COLOR_BGR2RGB)
                            pose_results = self.mediapipe_human_pose.process(body_roi)
                            face_results = self.mediapipe_face.process(body_roi)

                            #extract head . If head roi big enough, extract face descriptors.

                            mean_dist = 0.0
                            #for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            #    landmark_px = self.normalized_to_pixel_coordinates(landmark.x, landmark.y,
                            #                                                       roi_cols, roi_rows,
                            #                                                       int(xyxy[0]), int(xyxy[1]))
                            #mean_dist += depth_image.at(landmar_px.x, landmark_px.y)
                            #mean_dist /= len()

                            if self.display:
                                body_roi.flags.writeable = True
                                self.draw_landmarks(im0, body_roi, (int(xyxy[0]), int(xyxy[1])), pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                                if face_results.detections:
                                    for detection in face_results.detections:
                                        self.draw_detection(im0, body_roi, (int(xyxy[0]), int(xyxy[1])), detection)

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

    def draw_landmarks(self,
                       image: np.ndarray,
                       roi: np.ndarray,
                       roi_offset: Tuple[int, int],
                       landmark_list: landmark_pb2.NormalizedLandmarkList,
                       connections: Optional[List[Tuple[int, int]]] = None,
                       landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
                       connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

        if not landmark_list:
            return
        if image.shape[2] != 3:
            raise ValueError('Input image must contain three channel bgr data.')
        image_rows, image_cols, _ = image.shape
        roi_rows, roi_cols, _ = roi.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < _PRESENCE_THRESHOLD)):
                continue

            landmark_px = self.normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                               roi_cols, roi_rows,
                                                               roi_offset[0], roi_offset[1])
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                     f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    drawing_spec = connection_drawing_spec[connection] if isinstance(
                        connection_drawing_spec, Mapping) else connection_drawing_spec
                    cv2.line(image, idx_to_coordinates[start_idx],
                             idx_to_coordinates[end_idx], drawing_spec.color,
                             drawing_spec.thickness)
        # Draws landmark points after finishing the connection lines, which is
        # aesthetically better.
        if landmark_drawing_spec:
            for idx, landmark_px in idx_to_coordinates.items():
                drawing_spec = landmark_drawing_spec[idx] if isinstance(
                    landmark_drawing_spec, Mapping) else landmark_drawing_spec
                # White circle border
                circle_border_radius = max(drawing_spec.circle_radius + 1,
                                           int(drawing_spec.circle_radius * 1.2))
                cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                           drawing_spec.thickness)
                # Fill color into the circle
                cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                           drawing_spec.color, drawing_spec.thickness)

    def draw_detection(self,
                       image: np.ndarray,
                       roi: np.ndarray,
                       roi_offset: Tuple[int, int],
                       detection: detection_pb2.Detection,
                       keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=BLUE_COLOR),
                       bbox_drawing_spec: DrawingSpec = DrawingSpec(color=BLUE_COLOR)):
        """Draws the detction bounding box and keypoints on the image.

        Args:
          image: A three channel BGR image represented as numpy ndarray.
          detection: A detection proto message to be annotated on the image.
          keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
            drawing settings such as color, line thickness, and circle radius.
          bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
            drawing settings such as color and line thickness.

        Raises:
          ValueError: If one of the followings:
            a) If the input image is not three channel BGR.
            b) If the location data is not relative data.
        """
        if not detection.location_data:
            return
        if image.shape[2] != _BGR_CHANNELS:
            raise ValueError('Input image must contain three channel bgr data.')
        image_rows, image_cols, _ = roi.shape

        location = detection.location_data
        if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
            raise ValueError('LocationData must be relative for this drawing function to work.')
        # Draws keypoints.
        for keypoint in location.relative_keypoints:
            keypoint_px = self.normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                               image_cols, image_rows,
                                                               roi_offset[0], roi_offset[1])
            cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
                       keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
        # Draws bounding box if exists.
        if not location.HasField('relative_bounding_box'):
            return
        relative_bounding_box = location.relative_bounding_box
        rect_start_point = self.normalized_to_pixel_coordinates(relative_bounding_box.xmin,
                                                                relative_bounding_box.ymin,
                                                                image_cols, image_rows,
                                                                roi_offset[0], roi_offset[1])
        rect_end_point = self.normalized_to_pixel_coordinates(relative_bounding_box.xmin + relative_bounding_box.width,
                                                              relative_bounding_box.ymin + relative_bounding_box.height,
                                                              image_cols, image_rows,
                                                              roi_offset[0], roi_offset[1])
        cv2.rectangle(image, rect_start_point, rect_end_point, bbox_drawing_spec.color, bbox_drawing_spec.thickness)

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

    def letterbox(self, img: np.ndarray, new_shape=(640, 640),
                  color=(114, 114, 114),
                  auto: bool = True,
                  scaleFill: bool = False,
                  scaleup: bool = True,
                  stride: int = 32):

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

