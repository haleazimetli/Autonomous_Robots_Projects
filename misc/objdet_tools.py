import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils 
 
def plot_lidar_pcl(pcl):

    fig = plt.figure("LiDAR Point Cloud 3D", figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()

    ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=0.3, c=pcl[:, 2], cmap='viridis')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("LiDAR Point Cloud (3D)")
    ax.view_init(elev=20, azim=240)
    plt.pause(0.001)

def plot_camera_image(image):
    plt.imshow(image)
    plt.title("Camera Image")
    plt.axis('off')
    plt.show()


def create_bev_from_pcl(pcl, configs):
    mask = np.where((pcl[:, 0] >= configs.lim_x[0]) & (pcl[:, 0] <= configs.lim_x[1]) &
                    (pcl[:, 1] >= configs.lim_y[0]) & (pcl[:, 1] <= configs.lim_y[1]) &
                    (pcl[:, 2] >= configs.lim_z[0]) & (pcl[:, 2] <= configs.lim_z[1]))
    pcl = pcl[mask]

    bev_map = np.zeros((configs.bev_height, configs.bev_width))

    dx = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    dy = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width

    x_bev = ((pcl[:, 0] - configs.lim_x[0]) / dx).astype(np.int32)
    y_bev = ((pcl[:, 1] - configs.lim_y[0]) / dy).astype(np.int32)

    bev_map[x_bev, y_bev] = 1.0

    return bev_map

def plot_bev_map(bev_map, title="BEV Map"):
    plt.figure(figsize=(6, 6))
    plt.imshow(bev_map, cmap='gray', origin='lower')
    plt.title(title)
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.tight_layout()
    plt.show()

def show_bev_with_labels(frame, configs):

    lidar_name = dataset_pb2.LaserName.TOP
    lidar = utils.get(frame.lasers, lidar_name)
    calib = utils.get(frame.context.laser_calibrations, lidar_name)
    ri, cp, ri_pose = utils.parse_range_image_and_camera_projection(lidar)
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, cp, ri_pose, calib)

    bev_map = create_bev_from_pcl(pcl, configs)
    plot_bev_map(bev_map)

