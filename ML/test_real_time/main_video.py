""" folder_creator
This script tests the accuracy of the models.
It is used only as a testing tool to ensure that the training accuracy is accurate
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from tensorflow import keras
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers

img_height = 120
img_width = 160
sequence_length = 40
PERCENT = 25

actions = ['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out']

model_rgb = keras.Sequential(
    [
        layers.Conv3D(8, kernel_size=(3, 3, 4), input_shape=(40, 120, 160, 3), strides=(1, 1, 1), padding='valid',
                      activation='relu'),
        layers.MaxPool3D(),
        layers.Conv3D(16, 3, padding="same", activation="relu"),
        layers.MaxPool3D(),
        layers.BatchNormalization(),
        layers.Conv3D(8, 3, padding="same", activation="relu"),
        layers.MaxPool3D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(50, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(6, activation='softmax'),
    ]
)

model_depth = keras.Sequential(
    [
        layers.Conv3D(8, kernel_size=(3, 3, 4), input_shape=(40, 120, 160, 3), strides=(1, 1, 1), padding='valid',
                      activation='relu'),
        layers.MaxPool3D(),
        layers.Conv3D(16, 3, padding="same", activation="relu"),
        layers.MaxPool3D(),
        layers.BatchNormalization(),
        layers.Conv3D(8, 3, padding="same", activation="relu"),
        layers.MaxPool3D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(50, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(6, activation='softmax'),
    ]
)


def resize_image(img):
    width = int(img.shape[1] * PERCENT / 100)
    height = int(img.shape[0] * PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def video_tester():
    model_rgb.load_weights('video_rgb_weights.h5')
    model_depth.load_weights('video_depth_weights.h5')
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

    sequence_rgb = []
    predictions_rgb = []
    sequence_depth = []
    predictions_depth = []
    threshold = 0.7
    no_sequences = 10

    try:
        while True:
            nbrOfTakes = 0
            print("Press a to do a single prediction, b to do 10 in a row or press q to quit : ")
            action = input()
            if action == 'q':
                print('Quitting now...')
                return
            elif action == 'a':
                nbrOfTakes = 1
            elif action == 'b':
                nbrOfTakes = 10
            else:
                print('Wrong input please try again')
                continue
            input(f"Press enter to start the test")
            counter = 0
            pausePhase = False
            # Wait for a coherent pair of frames: depth and color
            while True:
                frames = pipeline.wait_for_frames()
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

                if pausePhase and len(sequence_rgb) == 20:
                    sequence_rgb = []
                    sequence_depth = []
                    pausePhase = False
                    print(f'Pause phase done, starting take of sample {counter}')

                if len(sequence_rgb) == sequence_length and len(sequence_depth) == sequence_length:
                    res_rgb = model_rgb.predict(np.expand_dims(sequence_rgb, axis=0))[0]
                    res_depth = model_depth.predict(np.expand_dims(sequence_depth, axis=0))[0]
                    predictions_rgb.append(np.argmax(res_rgb))
                    predictions_depth.append(np.argmax(res_depth))
                    # if np.argmax(np.bincount(predictions_rgb[-10:])) == np.argmax(res_rgb) and np.argmax(
                    #         np.bincount(predictions_depth[-10:])) == np.argmax(res_depth) and \
                    #         np.argmax(res_rgb) == np.argmax(res_depth):
                    #     if res_rgb[np.argmax(res_rgb)] > threshold and res_rgb[np.argmax(res_depth)] > threshold:
                    #         print(actions[np.argmax(res_rgb)])
                    print(
                        f"Prediction rgb : {actions[np.argmax(res_rgb)]}, prediction depth : {actions[np.argmax(res_depth)]}")
                    sequence_depth = []
                    sequence_rgb = []
                    counter += 1
                    if counter == nbrOfTakes:
                        break
                    else:
                        pausePhase = True
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    video_tester()
