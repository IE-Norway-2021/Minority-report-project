# import tf.lite as tf
import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
from tensorflow import keras
from tensorflow.keras import layers
import threading
import concurrent
from queue import Queue

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


def main_app_reduced_2_tflite():
    n = 3
    # Load TFLite model and allocate tensors.
    interpreter_rgb = tflite.Interpreter(model_path="video_rgb_reduced_2_pi_weights.tflite", num_threads=n)
    interpreter_depth = tflite.Interpreter(model_path="video_depth_reduced_2_pi_weights.tflite", num_threads=n)

    # Get input and output tensors.
    input_details_rgb = interpreter_rgb.get_input_details()
    output_details_rgb = interpreter_rgb.get_output_details()
    input_details_depth = interpreter_depth.get_input_details()
    output_details_depth = interpreter_depth.get_output_details()
    interpreter_rgb.allocate_tensors()
    interpreter_depth.allocate_tensors()

    # model_rgb.load_weights('video_rgb_reduced_2_weights.h5')
    # model_depth.load_weights('video_depth_reduced_2_weights.h5')
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
    threshold = 0.7
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
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.float32)

            color_image = resize_image(color_image)
            depth_image = resize_image(
                cv2.resize(np.asanyarray(colorizer.colorize(depth_frame).get_data(), np.float32), (640, 480),
                           interpolation=cv2.INTER_AREA))
            sequence_rgb.append(color_image)
            sequence_depth.append(depth_image)

            if len(sequence_rgb) >= nb_of_frames:
                sequence_depth = sequence_depth[-nb_of_frames:]
                sequence_rgb = sequence_rgb[-nb_of_frames:]
                # pred_rgb = model_rgb.predict(np.expand_dims(sequence_rgb, axis=0))[0]
                # pred_depth = model_depth.predict(np.expand_dims(sequence_depth, axis=0))[0]
                start = time.time()
                interpreter_rgb.set_tensor(input_details_rgb[0]['index'], [sequence_rgb])
                interpreter_depth.set_tensor(input_details_depth[0]['index'], [sequence_depth])
                interpreter_rgb.invoke()
                interpreter_depth.invoke()

                pred_rgb = interpreter_rgb.get_tensor(output_details_rgb[0]['index'])[0]
                pred_depth = interpreter_depth.get_tensor(output_details_depth[0]['index'])[0]
                end = time.time()
                print(end - start)

                print("the rgb output is {}".format(pred_rgb))
                print("the depth output is {}".format(pred_depth))

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
                            print(
                                f'Pred values : rgb={pred_rgb[np.argmax(pred_rgb)]}, depth={pred_depth[np.argmax(pred_depth)]}')
                            last_preds.clear()
                            sequence_rgb = sequence_rgb[-1:]
                            sequence_depth = sequence_depth[-1:]
                            print(f'all agreed it was a {actions[np.argmax(pred_rgb)]}')
                        else:
                            last_preds = last_preds[-(validation_num - 1):]

    finally:
        pipeline.stop()


input_shape = (20, 120, 160, 3)
model_rgb = keras.Sequential(  # pi version
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
model_depth = keras.Sequential(
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
threshold = 0.99

mutex = threading.Semaphore(1)
sequences_depth_queue = Queue()
sequences_rgb_queue = Queue()
mutex_preds = threading.Semaphore(1)
last_preds_queue = Queue()
threads_waiting = [0]
wait_lock = threading.Semaphore(0)
running = [1]


def thread_preduction_function(name):
    print(f'Thread {name} starting its execution...')
    while 1:
        mutex.acquire()
        if sequences_depth_queue.empty():
            threads_waiting[0] += 1
            mutex.release()
            print(f'Thread {name} starting to wait...')
            wait_lock.acquire()  # Passation de mutex lors du relachement
        if running[0] == 0:
            mutex.release()
            break
        sequence_depth = sequences_depth_queue.get()
        sequence_rgb = sequences_rgb_queue.get()
        mutex.release()
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
            mutex_preds.acquire()
            last_preds_queue.put(np.argmax(pred_depth))
            mutex_preds.release()
            print(
                f'Thread {name} found this prediction : {actions[np.argmax(pred_rgb)]} with accuracy rgb={pred_rgb[np.argmax(pred_rgb)]}, depth={pred_depth[np.argmax(pred_depth)]}')
    print(f'Thread {name} closing down...')


def main_app_reduced_2_tf():
    model_rgb.load_weights('video_rgb_reduced_2_pi_weights.h5')
    model_depth.load_weights('video_depth_reduced_2_pi_weights.h5')
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

    # Start streaming
    pipeline.start(config)

    frame_counter = 0
    sequence_rgb = []
    sequence_depth = []
    nb_of_frames = 20
    last_preds = []
    nbr_of_threads = 3
    validation_num = 4

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=nbr_of_threads) as executor:
            executor.map(thread_preduction_function, range(nbr_of_threads))
            time.sleep(2)
            print('Starting streaming...')
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
                    mutex.acquire()
                    sequences_depth_queue.put(sequence_depth)
                    sequences_rgb_queue.put(sequence_rgb)
                    if threads_waiting[0] > 0:
                        threads_waiting[0] -= 1
                        wait_lock.release()
                    else:
                        mutex.release()
                    mutex_preds.acquire()
                    while not last_preds_queue.empty():
                        last_preds.append(last_preds_queue.get())
                        mutex_preds.release()
                        if len(last_preds) >= validation_num:
                            result = all(elem == last_preds[0] for elem in last_preds)
                            if result:
                                print(f'all agreed it was a {actions[np.argmax(last_preds[0])]}')
                                mutex_preds.acquire()
                                while not last_preds_queue.empty(): #clear queue
                                    last_preds_queue.get()
                                mutex_preds.release()
                                sequence_rgb = []
                                sequence_depth = []
                            else:
                                last_preds.pop()
                        mutex_preds.acquire()
                    mutex_preds.release()


    finally:
        pipeline.stop()


if __name__ == '__main__':
    main_app_reduced_2_tf()
