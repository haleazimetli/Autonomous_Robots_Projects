# ---------------------------------------------------------------------
# Midterm Spring 2025 Exam
#
# UNCOMMENT THE EXERCISE PARTS AND COMPLETE THE MISSING CODE
# ----------------------------------------------------------------------

##################
# Imports:
# general packages

import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy
import zlib
from easydict import EasyDict as edict

# Add current working directory to path
sys.path.append(os.getcwd())

# Waymo open dataset reader to be installed under tools directory
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils

# misc. project-related imports
import misc.objdet_tools as tools
from  misc.objdet_tools import show_bev_with_labels
from misc.helpers import load_object_from_file

# add exercise directories to python path to enable relative imports
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
s2025 = 's2025'
mywork = 'mywork'
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, s2025)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, mywork)))

# import functions from individual exercise files
import s2025

##################
# Set parameters and perform initializations

# Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
show_only_frames = [50, 180]  # show only frames in interval for debugging

# set pause time between frames in ms (0 = stop between frames until key is pressed)
vis_pause_time = 0  

# Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename)  # adjustable path in case this script is called from another working directory
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator

##################
# Perform detection and evaluation of over all selected frames

cnt_frame = 0
det_performance_all = []  
while True:
    try:
        #################################
        # Get next frame from Waymo dataset

        frame = next(datafile_iter)
        if cnt_frame < show_only_frames[0]:
            cnt_frame = cnt_frame + 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('reached end of selected frames')
            break

        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        ####### START YOUR WORK #######
        #######

        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        
        # print no. of vehicles
        s2025.print_no_of_vehicles(frame) 

        # display camera image
        s2025.display_image(frame)

        # print angle of vertical field of view
        s2025.print_vfov_lidar(frame, lidar_name)

        # Load range image
        s2025.print_range_image_shape(frame, lidar_name)

        # Compute pitch angle resolution
        s2025.print_pitch_resolution(frame, lidar_name)

        # Retrieve maximum and minimum distance
        s2025.get_max_min_range(frame, lidar_name)

        # Visualize range channel
        s2025.vis_range_channel(frame, lidar_name)

        # Visualize intensity channel
        s2025.vis_intensity_channel(frame, lidar_name)

        # Convert range image to 3D point-cloud
        s2025.range_image_to_point_cloud(frame, lidar_name)

        # Define parameters used in subsequent steps
        configs = edict()
        configs.lim_x = [0, 50]
        configs.lim_y = [-25, 25]
        configs.lim_z = [-0.3, 3]
        configs.bev_width = 608
        configs.bev_height = 608
        configs.conf_thresh = 0.5
        configs.model = 'darknet'

        # Example C2-4-2 : count total no. of vehicles and vehicles that are difficult to track
        s2025.count_vehicles(frame)

        # Display label bounding boxes on top of BEV map
        show_bev_with_labels(frame, configs)
        
        det_performance = s2025.evaluate_frame_detections(frame)
        # Compute precision and recall (part 1/2 - remove comments only, no action inside functions required)
        det_performance_all.append(det_performance)  # store all evaluation results in a list for performance assessme

        #######
        
        # increment frame counter
        cnt_frame = cnt_frame + 1

    except StopIteration:
        # if StopIteration is raised, break from loop
        break

    # Exercise C2-4-5 : Compute precision and recall (part 2/2)
    s2025.compute_precision_recall(det_performance_all)

    # Plotting the precision-recall curve
    s2025.plot_precision_recall(det_performance_all, vis_pause_time)
