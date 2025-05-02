import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np

import matplotlib.pyplot as plt

from tools.waymo_reader.simple_waymo_open_dataset_reader import utils, dataset_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import label_pb2
from s2025.gt_projection import draw_projected_3d_box

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

def run_detector(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(image_rgb).to(device)
    with torch.no_grad():
        preds = model([tensor])[0]
    return preds

def draw_predicted_boxes(image, preds, score_thresh=0.5):
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if label.item() not in [3, 8]:  # 3 = car, 8 = truck
            continue
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        text = f"{label.item()} ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def extract_gt_boxes(frame):


    calib = utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
    vehicle_to_image = utils.get_image_transform(calib)
    boxes = []

    for label in frame.laser_labels:
        if label.type != label_pb2.Label.Type.TYPE_VEHICLE:
            continue
        box = label.box
        center = np.array([box.center_x, box.center_y, box.center_z, 1.0])
        proj = vehicle_to_image @ center
        if proj[2] <= 0:
            continue
        x_img = int(proj[0] / proj[2])
        y_img = int(proj[1] / proj[2])
        boxes.append({
            "label": "TYPE_VEHICLE",
            "bbox": [x_img - 20, y_img - 10, x_img + 20, y_img + 10]
        })
    return boxes


def draw_ground_truth_boxes(image, frame):
    calib = utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
    vehicle_to_image = utils.get_image_transform(calib)

    for label in frame.laser_labels:
        if label.type in [label_pb2.Label.Type.TYPE_VEHICLE]:
            utils.draw_3d_box(image, vehicle_to_image, label, colour=(0, 255, 0))
    
class MatplotlibVisualizer:
    def __init__(self, window_name="Detection Results", figsize=(12, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.im = self.ax.imshow(np.zeros((1280, 1920, 3), dtype=np.uint8))
        self.title = self.ax.set_title(window_name)
        self.ax.axis("off")
        self.saved = False  
        plt.ion()
        plt.show()

    def update(self, image, frame_id):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.im.set_data(image_rgb)
        self.title.set_text(f"Frame #{frame_id}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if not self.saved:
            self.save_last_frame(image)
            self.saved = True

    def save_last_frame(self, image, path="last_frame.png"):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imsave(path, image_rgb)

    def save_pr_curve(self, precision, recall, path="pr_curve.png"):
        plt.ioff()
        plt.figure(figsize=(6, 5))
        plt.plot([0, recall, 1], [1, precision, 0], marker='o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve (IoU ≥ 0.5)")
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()