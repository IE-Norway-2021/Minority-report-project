## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import time

import pyrealsense2 as rs
import numpy as np
from PIL import Image
import cv2
import os

# Path for exported data, numpy arrays
root_name = 'video_dataset'
# Actions that we try to detect
format = np.array(['rgb', 'depth'])
actions = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'start', 'stop'])
# Thirty videos worth of data
no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 70


# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
#
# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))
#
# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)
#
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#
# if device_product_line == 'L500':
#     config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#
# # Start streaming
# pipeline.start(config)
#
# try:
#     while True:
#
#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue
#
#         # Convert images to numpy arrays
#         # depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#
#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#
#         # depth_colormap_dim = depth_colormap.shape
#         # color_colormap_dim = color_image.shape
#         #
#         # # If depth and color resolutions are different, resize color image to match depth image for display
#         # if depth_colormap_dim != color_colormap_dim:
#         #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
#         #                                      interpolation=cv2.INTER_AREA)
#         #     images = np.hstack((resized_color_image, depth_colormap))
#         # else:
#         #     images = np.hstack((color_image, depth_colormap))
#
#         # Show images
#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', color_image)
#         cv2.waitKey(1)
#
# finally:
#
#     # Stop streaming
#     pipeline.stop()


def video_taker():
    #Setup pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    print('waiting for 30 frame')
    for x in range(30):
        pipeline.wait_for_frames()
    print('starting...')
    start = time.time()
    for frame_num in range(sequence_length):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)
        # Convert images to numpy arrays
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        cv2.imwrite(f'tmp/test_rgb{frame_num}.png', color_image)
        cv2.imwrite(f'tmp/test_depth{frame_num}.png', depth_image)
    now = time.time()
    end = now-start
    print(f'ended at {end:.5}s')


# sequence folder name 21.10.12-hh-mm_num
# NEW LOOP
# Loop through actions
# for action in actions:
#     # Loop through sequences aka videos
#     for sequence in range(no_sequences):
#         # Loop through video length aka sequence length
#         for frame_num in range(sequence_length):


if __name__ == '__main__':
    video_taker()
