import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils

from s2025.gt_projection import convert_gt_boxes_to_image_boxes

from s2025.detector_utils import run_detector, draw_predicted_boxes, draw_ground_truth_boxes, MatplotlibVisualizer

from s2025.metrics_q3 import evaluate_predictions, compute_precision_recall, plot_precision_recall, compute_average_precision

sys.path.append(os.getcwd())

os.makedirs("output", exist_ok=True)

filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
data_fullpath = os.path.join('dataset', filename)
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)

display_frame_range = [50, 180]
cnt_frame = 0
det_performance_all = []

visualizer = MatplotlibVisualizer()

plt.ion()  
fig, ax = plt.subplots(figsize=(12, 8))  
im = ax.imshow(np.zeros((1280,1920,3), dtype=np.uint8)) 
title = ax.set_title("Frame")
ax.axis("off")


while True:
    try:
        frame = next(datafile_iter)
        if cnt_frame < display_frame_range[0]:
            cnt_frame += 1
            continue
        elif cnt_frame > display_frame_range[1]:
            print("Reached end of selected frames")
            break

        print(f"Processing frame #{cnt_frame}")

        camera_name = dataset_pb2.CameraName.FRONT
        image = waymo_utils.get(frame.images, camera_name)
        img = waymo_utils.decode_image(image)

        preds = run_detector(img)
        gt_boxes = convert_gt_boxes_to_image_boxes(frame);
        det_performance = evaluate_predictions(preds, gt_boxes)
        det_performance_all.append(det_performance)

        draw_predicted_boxes(img, preds)
        draw_ground_truth_boxes(img, frame)
        visualizer.update(img, cnt_frame)

        cnt_frame += 1

    except StopIteration:
        break

precision, recall = compute_precision_recall(det_performance_all)

ap = compute_average_precision([recall], [precision]) 

print(f"\nPrecision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"mAP @ IoU 0.5: {ap:.3f}")

plot_precision_recall([precision], [recall], ap)

visualizer.save_last_frame(img, "output/last_frame.png")
visualizer.save_pr_curve(precision, recall, "output/pr_curve.png")