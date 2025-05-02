from PIL import Image
import io
import sys
import os
import cv2
import numpy as np
import zlib
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2

def load_range_image(frame, lidar_name):
    
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    return ri


def range_image_to_point_cloud(frame, lidar_name, vis=True):

    # extract range values from frame
    ri = load_range_image(frame, lidar_name)
    ri[ri<0]=0.0
    ri_range = ri[:,:,0]

    # load calibration data
    calibration = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]

    # compute vertical beam inclinations
    height = ri_range.shape[0]
    inclination_min = calibration.beam_inclination_min
    inclination_max = calibration.beam_inclination_max
    inclinations = np.linspace(inclination_min, inclination_max, height)
    inclinations = np.flip(inclinations)

    # compute azimuth angle and correct it so that the range image center is aligned to the x-axis
    width = ri_range.shape[1]
    extrinsic = np.array(calibration.extrinsic.transform).reshape(4,4)
    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
    azimuth = np.linspace(np.pi,-np.pi,width) - az_correction

    # expand inclination and azimuth such that every range image cell has its own appropriate value pair
    azimuth_tiled = np.broadcast_to(azimuth[np.newaxis,:], (height,width))
    inclination_tiled = np.broadcast_to(inclinations[:,np.newaxis],(height,width))

    # perform coordinate conversion
    x = np.cos(azimuth_tiled) * np.cos(inclination_tiled) * ri_range
    y = np.sin(azimuth_tiled) * np.cos(inclination_tiled) * ri_range
    z = np.sin(inclination_tiled) * ri_range

    # transform 3d points into vehicle coordinate system
    xyz_sensor = np.stack([x,y,z,np.ones_like(z)])
    xyz_vehicle = np.einsum('ij,jkl->ikl', extrinsic, xyz_sensor)
    xyz_vehicle = xyz_vehicle.transpose(1,2,0)

    # extract points with range > 0
    idx_range = ri_range > 0
    pcl = xyz_vehicle[idx_range,:3]
 
    # visualize point-cloud
    if vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        o3d.visualization.draw_geometries([pcd])

    # stack lidar point intensity as last column
    pcl_full = np.column_stack((pcl, ri[idx_range, 1]))    

    return pcl_full    


def vis_range_channel(frame, lidar_name):

    # extract range image from frame
    ri = load_range_image(frame, lidar_name)
    ri[ri<0]=0.0

    # map value range to 8bit
    ri_range = ri[:,:,0]
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)

    # focus on +/- 45° around the image center
    deg45 = int(img_range.shape[1] / 8)
    ri_center = int(img_range.shape[1]/2)
    img_range = img_range[:,ri_center-deg45:ri_center+deg45]

    print('max. val = ' + str(round(np.amax(img_range[:,:]),2)))
    print('min. val = ' + str(round(np.amin(img_range[:,:]),2)))

    cv2.imshow('range_image', img_range)
    cv2.waitKey(0)


def get_max_min_range(frame, lidar_name):

    # extract range image from frame
    ri = load_range_image(frame, lidar_name)
    ri[ri<0]=0.0

    print('max. range = ' + str(round(np.amax(ri[:,:,0]),2)) + 'm')
    print('min. range = ' + str(round(np.amin(ri[:,:,0]),2)) + 'm')


def print_range_image_shape(frame, lidar_name):

    ri = load_range_image(frame, lidar_name)
    print(ri.shape)

    # extract range data and convert to 8 bit
    #ri_range = ri[:,:,0]
    #ri_range = ri_range * 256 / (np.amax(ri_range) - np.amin(ri_range))
    #img_range = ri_range.astype(np.uint8)
    
    # visualize range image
    #cv2.imshow('range_image', img_range)
    #cv2.waitKey(0)


def print_vfov_lidar(frame, lidar_name):

    # get lidar calibration data
    calib_lidar = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]

    # compute vertical field of view (vfov) in rad
    vfov_rad = calib_lidar.beam_inclination_max - calib_lidar.beam_inclination_min

    # compute and print vfov in degrees
    print(vfov_rad*180/np.pi)


def display_image(frame):

    # load the camera data structure
    camera_name = dataset_pb2.CameraName.FRONT
    image = [obj for obj in frame.images if obj.name == camera_name][0]

    # convert the actual image into rgb format
    img = np.array(Image.open(io.BytesIO(image.image)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize the image to better fit the screen
    dim = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    resized = cv2.resize(img, dim)

    # display the image 
    cv2.imshow("Front-camera image", resized)
    cv2.waitKey(0)


def vis_intensity_channel(frame, lidar_name):

    ri = load_range_image(frame, lidar_name)
    ri[ri < 0] = 0.0
    intensity = ri[:,:,1]
    intensity = intensity * 255 / np.amax(intensity)
    img_intensity = intensity.astype(np.uint8)

    deg45 = int(img_intensity.shape[1] / 8)
    ri_center = int(img_intensity.shape[1]/2)
    img_intensity = img_intensity[:,ri_center-deg45:ri_center+deg45]

    cv2.imshow('intensity_image', img_intensity)
    cv2.waitKey(0)


def print_pitch_resolution(frame, lidar_name):

    ri = load_range_image(frame, lidar_name)
    height = ri.shape[0]
    calib_lidar = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    vfov_rad = calib_lidar.beam_inclination_max - calib_lidar.beam_inclination_min
    pitch_res_deg = (vfov_rad * 180 / np.pi) / height
    print('Pitch resolution =', round(pitch_res_deg * 60, 2), 'arcminutes')


def print_no_of_vehicles(frame):

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = len([label for label in frame.laser_labels if label.type == 1])
            
    print("number of labeled vehicles in current frame = " + str(num_vehicles))


