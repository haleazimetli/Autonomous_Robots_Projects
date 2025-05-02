import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box as shapely_box
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

from s2025.gt_projection import convert_gt_boxes_to_image_boxes
def compute_iou(boxA, boxB):
    b1 = shapely_box(*boxA)
    b2 = shapely_box(*boxB)
    inter = b1.intersection(b2).area
    union = b1.union(b2).area
    return inter / union if union > 0 else 0.0

def project_gt_boxes_to_image(frame):
    calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
    vehicle_to_image = waymo_utils.get_image_transform(calib)

    gt_boxes = []
    for label in frame.laser_labels:
        if label.type != label_pb2.Label.Type.TYPE_VEHICLE:
            continue
        box = label.box
        cx, cy, cz = box.center_x, box.center_y, box.center_z
        l, w, h = box.length, box.width, box.height
        yaw = box.heading

        corners = np.array([
            [ l/2,  w/2,  0],
            [ l/2, -w/2,  0],
            [-l/2, -w/2,  0],
            [-l/2,  w/2,  0],
            [ l/2,  w/2, -h],
            [ l/2, -w/2, -h],
            [-l/2, -w/2, -h],
            [-l/2,  w/2, -h]
        ])
        rot = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        translated = corners @ rot.T + np.array([cx, cy, cz])
        pts_homo = np.hstack((translated, np.ones((8, 1))))
        projected = (vehicle_to_image @ pts_homo.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        if np.any(np.isnan(projected)) or np.any(np.isinf(projected)):
            continue

        x1, y1 = projected[:, 0].min(), projected[:, 1].min()
        x2, y2 = projected[:, 0].max(), projected[:, 1].max()
        gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes

def evaluate_predictions(preds, gt_boxes, iou_threshold=0.5):
    tp = 0
    fp = 0
    matched_gt = set()

    pred_boxes = preds['boxes'].cpu().numpy()
    pred_labels = preds['labels'].cpu().numpy()
    pred_scores = preds['scores'].cpu().numpy()

    for pred_box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if label not in [3, 8] or score < 0.5:
            continue

        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return {'tp': tp, 'fp': fp, 'fn': fn}

def compute_precision_recall(results):
    tp = sum(d['tp'] for d in results)
    fp = sum(d['fp'] for d in results)
    fn = sum(d['fn'] for d in results)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall

def compute_mean_average_precision(precisions, recalls):
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_at_recall:
            ap += max(precisions_at_recall)
    ap /= 101
    return ap

def compute_average_precision(recalls, precisions):

    recall = np.array(recalls)
    precision = np.array(precisions)

    indices = np.argsort(recall)
    recall = recall[indices]
    precision = precision[indices]

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]

    return ap


def plot_precision_recall(precision, recall, ap, save_path="output/pr_curve.png"):
    plt.figure()
    plt.plot(recall, precision, marker='o')
    plt.title("PR Curve (IoU ≥ 0.5)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.annotate(f"AP = {ap:.2f}", xy=(0.6, 0.3), fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def print_final_metrics(precisions, recalls):

    ap = compute_average_precision(recalls, precisions)
    print(f"mAP @ IoU=0.5: {ap:.3f}")
    return ap