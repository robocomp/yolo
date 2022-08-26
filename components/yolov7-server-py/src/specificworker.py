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
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from typing import NamedTuple, List, Mapping, Optional, Tuple, Union
import dataclasses
from threading import Thread

# YOLOV7
#sys.path.append('/home/robocomp/software/yolov7')
sys.path.append('/home/robocomp/software/TensorRT-For-YOLO-Series')
sys.path.append('/home/robocomp/software/ONNX-YOLOv7-Object-Detection')
from YOLOv7 import YOLOv7

#from models.experimental import attempt_load
#from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier
#from utils.general import scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
#from utils.plots import plot_one_box
#from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.utils import preproc, vis
from utils.utils import BaseEngine

from mediapipe.framework.formats import landmark_pb2, detection_pb2, location_data_pb2
import mediapipe as mp
import queue

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

class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640, 640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz      # your model infer image size
        self.n_classes = 80     # your model classes


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
        self.Period = 1
        if startup_check:
            self.startup_check()
        else:
            self.frame = None
            self.opt = Options()
            #self.init_yolo_detect()
            self.init_yolo_trt()
            #print("Init_detect completed")
            self.yolov7_detector = YOLOv7('yolov7-w6-pose.onnx', conf_thres=0.5, iou_thres=0.5)

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            self.display = False
            self.detect_all = False

            # initialize  estimators
            # pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
            self.mediapipe_human_pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            # face
            #self.mp_face = mp.solutions.face_detection
            #self.mediapipe_face = self.mp_face.FaceDetection(min_detection_confidence=0.5)

            # queue
            self.input_queue = queue.Queue(1)
            self.output_queue = queue.Queue(1)

            self.timer.timeout.connect(self.compute)
            #self.timer.setSingleShot(True)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        self.display = params["display"] == "true" or params["display"] == "True"
        self.detect_all = params["detect_all"] == "true" or params["detect_all"] == "True"
        print("Params read. Starting...")
        return True

    @QtCore.Slot()
    def compute(self):
        # if self.detect_all:
        #     self.detect_objects_and_skeleton()
        # else:
        #     self.detect_skeleton()
        # self.detect_yolo()
        self.detect_yolo_onnxruntime()

        # FPS
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            print("Freq: ", self.cont, "Hz. Waiting for image")
            self.cont = 0
        else:
            self.cont += 1
        return True

#######################################################################################################
    def init_yolo_trt(self):
        self.pred = Predictor(engine_path='yolov7-tiny.trt')
        self.pred.get_fps()

    def init_yolo_detect(self, save_img=False):
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

        # Whatever
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
    
    def detect_yolo_onnxruntime(self):
         frame  = self.input_queue.get()
         boxes, scores, class_ids = self.yolov7_detector(frame)

         #combined_img = self.yolov7_detector.draw_detections(frame)
         #cv2.imshow("Detected Objects", combined_img)
         #cv2.waitKey(1)

         objects = []
         self.output_queue.put(objects)


    def detect_yolo(self):
        t0 = time.time()
        blob, ratio  = self.input_queue.get()
        t15 = time.time()
        #frame = np.frombuffer(rgb_image.image, dtype=np.uint8)
        #frame = frame.reshape((rgb_image.height, rgb_image.width, 3))
        #blob, ratio = preproc(frame, self.pred.imgsz, self.pred.mean, self.pred.std)

        t1= time.time()
        data = self.pred.infer(blob)
        t2 = time.time()
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

        objects = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                #if score < conf:
                #    continue
                
                # copy to interface 
                ibox = ifaces.RoboCompYoloServer.Box()
                ibox.name = self.pred.class_names[int(final_cls_inds[i])]
                ibox.prob = final_scores[i]
                ibox.left = int(box[0])
                ibox.top = int(box[1])
                ibox.right = int(box[2])
                ibox.bot = int(box[3])
                objects.append(ibox)

                # pose
                if final_cls_inds[i] == 0:  #person
                    t3 = time.time()
                    #lframe = frame.copy()
                    #body_roi = lframe[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                    #roi_rows, roi_cols, _ = body_roi.shape
                    #body_roi.flags.writeable = False
                    #pose_results = self.mediapipe_human_pose.process(body_roi)
                    t4 = time.time()    

            #img = frame
            #img = vis(img, final_boxes, final_scores, final_cls_inds,
            #            conf=0.5, class_names=self.pred.class_names)

        #cv2.imshow('frame', img)
        #cv2.waitKey(1)
        t5 = time.time()
        #print(1000.0*(t5-t0), 1000.0*(t15-t0), 1000.0*(t1-t0), 1000.0*(t2-t1), 1000.0*(t4-t3))
        self.output_queue.put(objects)

    def detect_skeleton(self):
        ext_image = self.input_queue.get()
        t0 = time.time()
        color = np.frombuffer(ext_image.image, dtype=np.uint8)
        color = color.reshape((ext_image.height, ext_image.width, 3))
        image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rows, cols, _ = image.shape
        image.flags.writeable = False
        t1 = time.time()
        pose_results = self.mediapipe_human_pose.process(image)
        t2 = time.time()

        if self.display:
            self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Jetson", image)
            cv2.waitKey(1)  # 1 millisecond

        objects = []
        self.output_queue.put(objects)    # synchronize with interface
        t3 = time.time()
        #print(f'Total {(1E3 * (t3 - t0)):.1f}ms, Inference {(1E3 * (t2 - t1)):.1f}ms')

    def detect_objects_and_skeleton(self):
        # ext_image = self.input_queue.get()
        # t0 = time.time()
        # color = np.frombuffer(ext_image.image, dtype=np.uint8)
        # color = color.reshape((ext_image.height, ext_image.width, 3))
        # #depth = np.frombuffer(self.ext_image.depth, dtype=np.float32)
        # img, ratio, (dw, dh) = self.letterbox(color)
        #
        # # Convert img format
        # img = np.stack([img], 0)
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img).to(self.device)
        # img = img.half() if self.half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        #
        # # Inference
        # t1 = time.time()
        # pred = self.model(img, augment=self.opt.augment)[0]
        # t2 = time.time()
        #
        # # Apply NMS
        # pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,  #classes=self.opt.classes,#
        #                            agnostic=self.opt.agnostic_nms)
        # t3 = time.time()
        #
        # # Process detections
        # objects = []   # use a double buffer to avoid deleting before read in interface
        # if len(pred):
        #     det = pred[0]  # detections per image
        #     im0 = color
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        #
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if cls == 0:  # if person extract skeleton and face win Mediapipe
        #                 t4 = time_synchronized()
        #                 body_roi = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
        #                 roi_rows, roi_cols, _ = body_roi.shape
        #                 body_roi.flags.writeable = False
        #                 body_roi = cv2.cvtColor(body_roi, cv2.COLOR_BGR2RGB)
        #                 pose_results = self.mediapipe_human_pose.process(body_roi)
        #                 face_results = self.mediapipe_face.process(body_roi)
        #                 t5 = time_synchronized()
        #                 mean_dist = 0.0
        #                 #for idx, landmark in enumerate(results.pose_landmarks.landmark):
        #                 #    landmark_px = self.normalized_to_pixel_coordinates(landmark.x, landmark.y,
        #                 #                                                       roi_cols, roi_rows,
        #                 #                                                       int(xyxy[0]), int(xyxy[1]))
        #                 #mean_dist += depth_image.at(landmar_px.x, landmark_px.y)
        #                 #mean_dist /= len()
        #
        #                 if self.display:
        #                     self.draw_landmarks(im0, body_roi, (int(xyxy[0]), int(xyxy[1])), pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        #                     if face_results.detections:
        #                         self.draw_detection(im0, body_roi, (int(xyxy[0]), int(xyxy[1])), face_results.detections[0])
        #
        #             if self.display:  # Add bbox to image
        #                 label = f'{self.names[int(cls)]} {conf:.2f}'
        #                 plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
        #
        #             # copy to interface data
        #             box = ifaces.RoboCompYoloServer.Box()
        #             box.name = self.names[int(cls)]
        #             box.prob = float(conf)
        #             box.left = int(xyxy[0])
        #             box.top = int(xyxy[1])
        #             box.right = int(xyxy[2])
        #             box.bot = int(xyxy[3])
        #             objects.append(box)
        # t6 = time_synchronized()
        # # Print time (inference + NMS)
        # print(f'Total {(1E3 * (t4 - t0)):.1f}ms, Image {(1E3 * (t1 - t0)):.1f}ms, Inference {(1E3 * (t2 - t1)):.1f}ms, '
        #       f'NMS {(1E3 * (t3 - t2)):.1f}ms, Pose {(1E3 * (t5 - t4)):.1f}ms, Drawing {(1E3 * (t6 - t5)):.1f}ms')
        # self.output_queue.put(objects)    # synchronize with interface
        pass

    def detect_objects(self):
        # ext_image = self.input_queue.get()
        # t0 = time_synchronized()
        # color = np.frombuffer(ext_image.image, dtype=np.uint8)
        # color = color.reshape((ext_image.height, ext_image.width, 3))
        # # depth = np.frombuffer(self.ext_image.depth, dtype=np.float32)
        # img, ratio, (dw, dh) = self.letterbox(color)
        #
        # # Convert img format
        # img = np.stack([img], 0)
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img).to(self.device)
        # img = img.half() if self.half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        #
        # # Warmup
        # # if self.device.type != 'cpu' and (
        # #         old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        # #     old_img_b = img.shape[0]
        # #     old_img_h = img.shape[2]
        # #     old_img_w = img.shape[3]
        # #     for i in range(3):
        # #         model(img, augment=self.opt.augment)[0]
        #
        # # Inference
        # t1 = time_synchronized()
        # pred = self.model(img, augment=self.opt.augment)[0]
        # t2 = time_synchronized()
        #
        # # Apply NMS
        # pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,  # classes=self.opt.classes,#
        #                            agnostic=self.opt.agnostic_nms)
        # t3 = time_synchronized()
        #
        # # Process detections
        # objects = []  # use a double buffer to avoid deleting before read in interface
        # if len(pred):
        #     det = pred[0]  # detections per image
        #     im0 = color.copy()
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        #
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if self.display:  # Add bbox to image
        #                 label = f'{self.names[int(cls)]} {conf:.2f}'
        #                 plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
        #
        #             # copy to interface data
        #             box = ifaces.RoboCompYoloServer.Box()
        #             box.name = self.names[int(cls)]
        #             box.prob = float(conf)
        #             box.left = int(xyxy[0])
        #             box.top = int(xyxy[1])
        #             box.right = int(xyxy[2])
        #             box.bot = int(xyxy[3])
        #             objects.append(box)
        #
        # # Print time (inference + NMS)
        # # print(f'Total {(1E3 * (t2 - t0)):.1f}ms, Inference {(1E3 * (t2 - t1)):.1f}ms, NMS {(1E3 * (t3 - t2)):.1f}ms')
        #
        #
        # if self.display:
        #     cv2.imshow("Jetson", im0)
        #     cv2.waitKey(1)  # 1 millisecond
        #
        # self.output_queue.put(objects)    # synchronize with interface
        pass

    def draw_landmarks(self,
                       image: np.ndarray,
                       roi: np.ndarray,
                       roi_offset: Tuple[int, int],
                       landmark_list: landmark_pb2.NormalizedLandmarkList,
                       connections: Optional[List[Tuple[int, int]]] = None,
                       landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
                       connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

        """Draws the landmarks and the connections on the image.

          Args:
            image: A three channel BGR image represented as numpy ndarray.
            landmark_list: A normalized landmark list proto message to be annotated on
              the image.
            connections: A list of landmark index tuples that specifies how landmarks to
              be connected in the drawing.
            landmark_drawing_spec: Either a DrawingSpec object or a mapping from
              hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
              settings such as color, line thickness, and circle radius.
              If this argument is explicitly set to None, no landmarks will be drawn.
            connection_drawing_spec: Either a DrawingSpec object or a mapping from
              hand connections to the DrawingSpecs that specifies the
              connections' drawing settings such as color and line thickness.
              If this argument is explicitly set to None, no landmark connections will
              be drawn.

          Raises:
            ValueError: If one of the followings:
              a) If the input image is not three channel BGR.
              b) If any connetions contain invalid landmark index.
          """

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
        frame = np.frombuffer(img.image, dtype=np.uint8)
        frame = frame.reshape((img.height, img.width, 3))
        #blob, ratio = preproc(frame, self.pred.imgsz, self.pred.mean, self.pred.std)
        #self.input_queue.put([blob,ratio])
        self.input_queue.put(frame)
        return self.output_queue.get()

    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompYoloServer you can use this types:
    # RoboCompYoloServer.TImage
    # RoboCompYoloServer.Box


