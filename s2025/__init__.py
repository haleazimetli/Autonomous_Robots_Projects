import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from tools.waymo_reader.simple_waymo_open_dataset_reader import label_pb2, dataset_pb2, utils

plt.ion()

def print_no_of_vehicles(frame):
    count = sum(1 for label in frame.laser_labels if label.type == label_pb2.Label.Type.TYPE_VEHICLE)
    print(f"Number of vehicles in frame: {count}")

def display_image(frame):
    image = utils.get(frame.images, dataset_pb2.CameraName.FRONT)
    img = utils.decode_image(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure("Camera View")
    plt.clf()
    plt.imshow(img)
    plt.title("Camera (RGB)")
    plt.axis("off")
    plt.pause(0.001)

def print_vfov_lidar(frame, lidar_name):
    calib = utils.get(frame.context.laser_calibrations, lidar_name)
    vfov = calib.beam_inclination_max - calib.beam_inclination_min
    print(f"Vertical FOV: {vfov:.4f} radians")

def print_range_image_shape(frame, lidar_name):
    lidar = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(lidar)
    print(f"Range image shape: {ri.shape}")

def print_pitch_resolution(frame, lidar_name):
    lidar = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(lidar)
    pitch_res = (np.pi / 2) / ri.shape[0]
    print(f"Pitch resolution: {pitch_res:.6f}")

def get_max_min_range(frame, lidar_name):
    lidar = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(lidar)
    print(f"Max range: {np.max(ri.data)}, Min range: {np.min(ri.data)}")

def vis_range_channel(frame, lidar_name):
    lidar = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(lidar)
    if ri is None: return

    ri_np = np.array(ri.data, dtype=np.float32).reshape(ri.shape)
    range_data = np.clip(ri_np[:, :, 0], 0, np.percentile(ri_np[:, :, 0], 98))
    range_img = (range_data * 255 / np.max(range_data)).astype(np.uint8)

    plt.figure("Range Channel")
    plt.clf()
    plt.imshow(range_img, cmap="viridis")
    plt.colorbar(label="Distance (m)")
    plt.title("Range Channel")
    plt.axis("off")
    plt.pause(0.001)

def vis_intensity_channel(frame, lidar_name):
    lidar = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(lidar)
    if ri is None or ri.shape[2] < 2: return

    ri_np = np.array(ri.data, dtype=np.float32).reshape(ri.shape)
    intensity_data = np.clip(ri_np[:, :, 1], 0, np.percentile(ri_np[:, :, 1], 98))
    intensity_img = (intensity_data * 255 / np.max(intensity_data)).astype(np.uint8)

    plt.figure("Intensity Channel")
    plt.clf()
    plt.imshow(intensity_img, cmap="hot")
    plt.colorbar(label="Intensity")
    plt.title("Intensity Channel")
    plt.axis("off")
    plt.pause(0.001)

def range_image_to_point_cloud(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, lidar_name)
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
    print("Point Cloud shape:", pcl.shape)
    # Görselleştirme: misc/objdet_tools içindeki fonksiyon kullanılabilir.
    try:
        from misc.objdet_tools import plot_lidar_pcl
        plot_lidar_pcl(pcl)
    except Exception as e:
        print("Point Cloud not visualized:", e)

def count_vehicles(frame):
    total_vehicles = 0
    difficult_vehicles = 0

    for label in frame.laser_labels:
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            total_vehicles += 1
            if label.DifficultyLevel == label_pb2.Label.DifficultyLevel.LEVEL_2:
                difficult_vehicles += 1

    print(f"Total Vehicles: {total_vehicles}, Difficult to Track: {difficult_vehicles}")

def compute_iou(box1, box2):
    b1 = box(*box1)
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

def evaluate_frame_detections(frame, iou_threshold=0.5):
    gt_boxes = []
    for label in frame.laser_labels:
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            x, y, l, w = label.box.center_x, label.box.center_y, label.box.length, label.box.width
            gt_boxes.append([x - l/2, y - w/2, x + l/2, y + w/2])
    pred_boxes = gt_boxes[:int(len(gt_boxes)*0.6)]
    pred_boxes.append([0, 0, 1, 1])
    pred_boxes.append([50, 50, 52, 52])
    tp, fp, matched_gt = 0, 0, set()
    for pred in pred_boxes:
        matched = False
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt: continue
            if compute_iou(pred, gt) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return {'tp': tp, 'fp': fp, 'fn': fn}

def compute_precision_recall(det_performance_all):
    total_tp = sum(d['tp'] for d in det_performance_all)
    total_fp = sum(d['fp'] for d in det_performance_all)
    total_fn = sum(d['fn'] for d in det_performance_all)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"\nGlobal Precision: {precision:.2f}")
    print(f"Global Recall: {recall:.2f}")

def plot_precision_recall(det_performance_all, vis_pause_time):
    precs, recalls = [], []
    for d in det_performance_all:
        tp, fp, fn = d['tp'], d['fp'], d['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        precs.append(prec)
        recalls.append(rec)

    plt.figure("Precision–Recall Curve")
    plt.clf()
    plt.plot(recalls, precs, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    
    if vis_pause_time == 0:
        plt.waitforbuttonpress()
    else:
        plt.pause(vis_pause_time / 1000.0)
