import numpy as np
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils, dataset_pb2, label_pb2
import cv2

def convert_gt_boxes_to_image_boxes(frame):
    calib = utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
    vehicle_to_image = utils.get_image_transform(calib)
    boxes = []

    for label in frame.laser_labels:
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            corners = utils.get_3d_box_projected_corners(vehicle_to_image, label)
            if corners is not None:
                x1, y1, x2, y2 = utils.compute_2d_bounding_box((1280, 1920), corners)
                boxes.append([x1, y1, x2, y2])
    return boxes

def draw_projected_3d_box(image, box, calib):

    vehicle_to_image = utils.get_image_transform(calib)
    
    corners_3d = utils.compute_box_corners_3d(box)  # returns (8, 3)
    corners_hom = np.hstack((corners_3d, np.ones((8, 1))))  # (8, 4)
    
    projected = (vehicle_to_image @ corners_hom.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    for i in range(4):
        pt1 = tuple(projected[i].astype(int))
        pt2 = tuple(projected[(i+1)%4].astype(int))
        pt3 = tuple(projected[i+4].astype(int))
        pt4 = tuple(projected[(i+1)%4 + 4].astype(int))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        cv2.line(image, pt1, pt3, (0, 255, 0), 2)
        cv2.line(image, pt3, pt4, (0, 255, 0), 2)