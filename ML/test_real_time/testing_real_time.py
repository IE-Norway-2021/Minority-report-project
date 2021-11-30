import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model

img_height = 120
img_width = 160
sequence_length = 10
PERCENT = 25
rate = 30

actions = ['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out']


def resize_image(img):
    width = int(img.shape[1] * PERCENT / 100)
    height = int(img.shape[0] * PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def main_app_reduced_4():
    model_reduced_4_beggining_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(3, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_4_beggining_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(3, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_4_middle_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(4, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_4_middle_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(4, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_4_end_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(3, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_4_end_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(3, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]

    )
    model_reduced_4_beggining_rgb.load_weights('video_rgb_reduced_beginning_weights.h5')
    model_reduced_4_beggining_depth.load_weights('video_depth_reduced_beginning_weights.h5')
    model_reduced_4_middle_rgb.load_weights('video_rgb_reduced_middle_weights.h5')
    model_reduced_4_middle_depth.load_weights('video_depth_reduced_middle_weights.h5')
    model_reduced_4_end_rgb.load_weights('video_rgb_reduced_end_weights.h5')
    model_reduced_4_end_depth.load_weights('video_depth_reduced_end_weights.h5')
    print('Finished loading models')
    # Configure depth and color streams
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

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    print('Starting streaming...')

    # Start streaming
    pipeline.start(config)

    frame_counter = 0
    sequence_rgb = []
    sequence_depth = []
    threshold = 0.7

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                continue
            frame_counter = (frame_counter + 1) % 4
            if frame_counter != 0:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            color_image = resize_image(color_image)
            depth_image = resize_image(
                cv2.resize(np.asanyarray(colorizer.colorize(depth_frame).get_data()), (640, 480),
                           interpolation=cv2.INTER_AREA))
            sequence_rgb.append(color_image)
            sequence_depth.append(depth_image)

            if len(sequence_rgb) >= 10:
                sequence_depth = sequence_depth[-10:]
                sequence_rgb = sequence_rgb[-10:]
                sequence_rgb_beginning = sequence_rgb[0:3]
                sequence_rgb_middle = sequence_rgb[3:7]
                sequence_rgb_end = sequence_rgb[7:10]
                sequence_depth_beginning = sequence_depth[0:3]
                sequence_depth_middle = sequence_depth[3:7]
                sequence_depth_end = sequence_depth[7:10]
                pred_beg_rgb = model_reduced_4_beggining_rgb.predict(np.expand_dims(sequence_rgb_beginning, axis=0))[0]
                pred_mid_rgb = model_reduced_4_middle_rgb.predict(np.expand_dims(sequence_rgb_middle, axis=0))[0]
                pred_end_rgb = model_reduced_4_end_rgb.predict(np.expand_dims(sequence_rgb_end, axis=0))[0]
                pred_beg_depth = \
                    model_reduced_4_beggining_depth.predict(np.expand_dims(sequence_depth_beginning, axis=0))[0]
                pred_mid_depth = model_reduced_4_middle_depth.predict(np.expand_dims(sequence_depth_middle, axis=0))[0]
                pred_end_depth = model_reduced_4_end_rgb.predict(np.expand_dims(sequence_depth_end, axis=0))[0]
                # if (np.argmax(pred_beg_rgb) == np.argmax(pred_beg_depth)) and (
                #         np.argmax(pred_mid_rgb) == np.argmax(pred_mid_depth)) and (
                #         np.argmax(pred_end_rgb) == np.argmax(pred_end_depth)) and (
                #         np.argmax(pred_beg_rgb) == np.argmax(pred_mid_rgb)) and (
                #         np.argmax(pred_beg_rgb) == np.argmax(pred_end_rgb)):
                #     all_valid = True
                #     for test_res in [pred_beg_rgb, pred_mid_rgb, pred_end_rgb, pred_beg_depth, pred_mid_depth,
                #                      pred_end_depth]:
                #         if test_res[np.argmax(test_res)] < threshold:
                #             all_valid = False
                #             break
                #     if not all_valid:
                #         continue
                #     # all good, can send to driver!
                #     print(f'all agreed it was a {actions[np.argmax(pred_beg_rgb)]}')
                if (np.argmax(pred_beg_rgb) == np.argmax(pred_mid_rgb)) and (
                        np.argmax(pred_beg_rgb) == np.argmax(pred_end_rgb)):
                    print(f'all rgb agreed it was a {actions[np.argmax(pred_beg_rgb)]}')
                if (np.argmax(pred_beg_depth) == np.argmax(pred_mid_depth)) and (
                        np.argmax(pred_beg_depth) == np.argmax(pred_end_depth)):
                    print(f'all depth agreed it was a {actions[np.argmax(pred_beg_depth)]}')

    finally:
        pipeline.stop()


def main_app_reduced_2():
    model_reduced_2_beggining_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(6, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_2_beggining_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(6, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_2_middle_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(8, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_2_middle_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(8, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_2_end_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(6, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_reduced_2_end_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(6, 120, 160, 3), strides=(1, 1, 1),
                          padding='same',
                          activation='relu'),
            layers.MaxPool3D(padding="same"),
            layers.Conv3D(32, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Conv3D(16, 1, padding="same", activation="relu"),
            layers.MaxPool3D(padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )
    model_reduced_2_beggining_rgb.load_weights('video_rgb_reduced_2_beginning_weights.h5')
    model_reduced_2_beggining_depth.load_weights('video_depth_reduced_2_beginning_weights.h5')
    model_reduced_2_middle_rgb.load_weights('video_rgb_reduced_2_middle_weights.h5')
    model_reduced_2_middle_depth.load_weights('video_depth_reduced_2_middle_weights.h5')
    model_reduced_2_end_rgb.load_weights('video_rgb_reduced_2_end_weights.h5')
    model_reduced_2_end_depth.load_weights('video_depth_reduced_2_end_weights.h5')
    print('Finished loading models')
    # Configure depth and color streams
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

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    print('Starting streaming...')

    # Start streaming
    pipeline.start(config)

    frame_counter = 0
    sequence_rgb = []
    sequence_depth = []
    threshold = 0.5
    nb_of_frames = 20

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                continue
            frame_counter = (frame_counter + 1) % 2
            if frame_counter != 0:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            color_image = resize_image(color_image)
            depth_image = resize_image(
                cv2.resize(np.asanyarray(colorizer.colorize(depth_frame).get_data()), (640, 480),
                           interpolation=cv2.INTER_AREA))
            sequence_rgb.append(color_image)
            sequence_depth.append(depth_image)

            if len(sequence_rgb) >= nb_of_frames:
                sequence_depth = sequence_depth[-nb_of_frames:]
                sequence_rgb = sequence_rgb[-nb_of_frames:]
                sequence_rgb_beginning = sequence_rgb[0:6]
                sequence_rgb_middle = sequence_rgb[6:14]
                sequence_rgb_end = sequence_rgb[14:20]
                sequence_depth_beginning = sequence_depth[0:6]
                sequence_depth_middle = sequence_depth[6:14]
                sequence_depth_end = sequence_depth[14:20]
                pred_beg_rgb = model_reduced_2_beggining_rgb.predict(np.expand_dims(sequence_rgb_beginning, axis=0))[0]
                pred_mid_rgb = model_reduced_2_middle_rgb.predict(np.expand_dims(sequence_rgb_middle, axis=0))[0]
                pred_end_rgb = model_reduced_2_end_rgb.predict(np.expand_dims(sequence_rgb_end, axis=0))[0]
                pred_beg_depth = \
                    model_reduced_2_beggining_depth.predict(np.expand_dims(sequence_depth_beginning, axis=0))[0]
                pred_mid_depth = model_reduced_2_middle_depth.predict(np.expand_dims(sequence_depth_middle, axis=0))[0]
                pred_end_depth = model_reduced_2_end_rgb.predict(np.expand_dims(sequence_depth_end, axis=0))[0]
                # if (np.argmax(pred_beg_rgb) == np.argmax(pred_beg_depth)) and (
                #         np.argmax(pred_mid_rgb) == np.argmax(pred_mid_depth)) and (
                #         np.argmax(pred_end_rgb) == np.argmax(pred_end_depth)) and (
                #         np.argmax(pred_beg_rgb) == np.argmax(pred_mid_rgb)) and (
                #         np.argmax(pred_beg_rgb) == np.argmax(pred_end_rgb)):
                #     all_valid = True
                #     for test_res in [pred_beg_rgb, pred_mid_rgb, pred_end_rgb, pred_beg_depth, pred_mid_depth,
                #                      pred_end_depth]:
                #         if test_res[np.argmax(test_res)] < threshold:
                #             all_valid = False
                #             break
                #     if not all_valid:
                #         continue
                #     # all good, can send to driver!
                #     print(f'all agreed it was a {actions[np.argmax(pred_beg_rgb)]}')
                all_valid = True
                for test_res in [pred_beg_rgb, pred_mid_rgb, pred_end_rgb, pred_beg_depth, pred_mid_depth,
                                 pred_end_depth]:
                    if test_res[np.argmax(test_res)] < threshold:
                        all_valid = False
                        break
                if not all_valid:
                    continue
                if (np.argmax(pred_beg_rgb) == np.argmax(pred_mid_rgb)) and (
                        np.argmax(pred_beg_rgb) == np.argmax(pred_end_rgb)):
                    print(f'all rgb agreed it was a {actions[np.argmax(pred_beg_rgb)]}')
                if (np.argmax(pred_beg_depth) == np.argmax(pred_mid_depth)) and (
                        np.argmax(pred_beg_depth) == np.argmax(pred_end_depth)):
                    print(f'all depth agreed it was a {actions[np.argmax(pred_beg_depth)]}')

    finally:
        pipeline.stop()


def main_app_reduced_4_full():
    model_rgb = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(10, 120, 160, 3), strides=(1, 1, 1),
                          padding='valid',
                          activation='relu'),
            layers.MaxPool3D(),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_depth = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(10, 120, 160, 3), strides=(1, 1, 1),
                          padding='valid',
                          activation='relu'),
            layers.MaxPool3D(),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(120, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )
    model_rgb.load_weights('video_rgb_reduced_4_weights.h5')
    model_depth.load_weights('video_depth_reduced_4_weights.h5')
    print('Finished loading models')
    # Configure depth and color streams
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, rate)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, rate)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, rate)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    print('Starting streaming...')

    # Start streaming
    pipeline.start(config)

    frame_counter = 0
    sequence_rgb = []
    sequence_depth = []
    threshold = 0.99999
    nb_of_frames = 10
    last_preds = []
    validation_num = 4

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                continue
            frame_counter = (frame_counter + 1) % 4
            if frame_counter != 0:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            color_image = resize_image(color_image)
            depth_image = resize_image(
                cv2.resize(np.asanyarray(colorizer.colorize(depth_frame).get_data()), (640, 480),
                           interpolation=cv2.INTER_AREA))
            sequence_rgb.append(color_image)
            sequence_depth.append(depth_image)

            if len(sequence_rgb) >= nb_of_frames:
                sequence_depth = sequence_depth[-nb_of_frames:]
                sequence_rgb = sequence_rgb[-nb_of_frames:]
                pred_rgb = model_rgb.predict(np.expand_dims(sequence_rgb, axis=0))[0]
                pred_depth = model_depth.predict(np.expand_dims(sequence_depth, axis=0))[0]
                all_valid = True
                for test_res in [pred_rgb, pred_depth]:
                    if test_res[np.argmax(test_res)] < threshold:
                        all_valid = False
                        break
                if not all_valid:
                    continue
                if np.argmax(pred_rgb) == np.argmax(pred_depth):
                    last_preds.append(np.argmax(pred_depth))
                    if len(last_preds) >= 3:
                        result = all(elem == last_preds[0] for elem in last_preds)
                        if result:
                            print(f'Pred values : rgb={pred_rgb[np.argmax(pred_rgb)]}, depth={pred_depth[np.argmax(pred_depth)]}')
                            last_preds.clear()
                            sequence_rgb = sequence_rgb[-1:]
                            sequence_depth = sequence_depth[-1:]
                            print(f'all agreed it was a {actions[np.argmax(pred_rgb)]}')
                        else:
                            last_preds = last_preds[-(validation_num-1):]

    finally:
        pipeline.stop()


def main_app_reduced_2_full():
    model_rgb_old = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(20, 120, 160, 3), strides=(1, 1, 1),
                          padding='valid',
                          activation='relu'),
            layers.MaxPool3D(),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    model_depth_old = keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(20, 120, 160, 3), strides=(1, 1, 1),
                          padding='valid',
                          activation='relu'),
            layers.MaxPool3D(),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Conv3D(16, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )

    input_shape = (20,120,160,3)
    model_rgb_semi_opti =  keras.Sequential( #pi version
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=input_shape, strides=(1, 1, 1),
                            padding='valid', activation='relu'),
            layers.MaxPool3D(),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(80, activation='relu'),
            layers.Dense(40, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )
    model_depth_semi_opti =  keras.Sequential(
        [
            layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=input_shape, strides=(1, 1, 1),
                            padding='valid', activation='relu'),
            layers.MaxPool3D(),
            layers.Conv3D(32, 3, padding="same", activation="relu"),
            layers.MaxPool3D(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(80, activation='relu'),
            layers.Dense(40, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax'),
        ]
    )
    vid_input = Input(input_shape)
    x = Conv3D(16, kernel_size=(3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu')(vid_input)
    x = MaxPooling3D(padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(32, kernel_size=(3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = MaxPooling3D(padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(16, kernel_size=(3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = MaxPooling3D(padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(60, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(6, activation='softmax')(x)
    model_rgb = Model(vid_input, x, name='rgb')
    model_depth = Model(vid_input, x, name='depth')
    model_rgb.load_weights('video_rgb_reduced_2_weights.h5')
    model_depth.load_weights('video_depth_reduced_2_weights.h5')
    print('Finished loading models')
    # Configure depth and color streams
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, rate)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, rate)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, rate)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    print('Starting streaming...')

    # Start streaming
    pipeline.start(config)

    frame_counter = 0
    sequence_rgb = []
    sequence_depth = []
    threshold = 0.9
    nb_of_frames = 20
    last_preds = []
    validation_num = 6

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                continue
            frame_counter = (frame_counter + 1) % 2
            if frame_counter != 0:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            color_image = resize_image(color_image)
            depth_image = resize_image(
                cv2.resize(np.asanyarray(colorizer.colorize(depth_frame).get_data()), (640, 480),
                           interpolation=cv2.INTER_AREA))
            sequence_rgb.append(color_image)
            sequence_depth.append(depth_image)

            if len(sequence_rgb) >= nb_of_frames:
                sequence_depth = sequence_depth[-nb_of_frames:]
                sequence_rgb = sequence_rgb[-nb_of_frames:]
                start = time.time()
                pred_rgb = model_rgb.predict(np.expand_dims(sequence_rgb, axis=0))[0]
                pred_depth = model_depth.predict(np.expand_dims(sequence_depth, axis=0))[0]
                end = time.time()
                print(end-start)
                all_valid = True
                for test_res in [pred_rgb, pred_depth]:
                    if test_res[np.argmax(test_res)] < threshold:
                        all_valid = False
                        break
                if not all_valid:
                    continue
                if np.argmax(pred_rgb) == np.argmax(pred_depth):
                    last_preds.append(np.argmax(pred_depth))
                    if len(last_preds) >= validation_num:
                        result = all(elem == last_preds[0] for elem in last_preds)
                        if result:
                            print(f'Pred values : rgb={pred_rgb[np.argmax(pred_rgb)]}, depth={pred_depth[np.argmax(pred_depth)]}')
                            last_preds.clear()
                            sequence_rgb = sequence_rgb[-1:]
                            sequence_depth = sequence_depth[-1:]
                            print(f'all agreed it was a {actions[np.argmax(pred_rgb)]}')
                        else:
                            last_preds = last_preds[-(validation_num-1):]

    finally:
        pipeline.stop()


if __name__ == '__main__':
    main_app_reduced_2_full()
