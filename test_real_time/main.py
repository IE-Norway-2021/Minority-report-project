import cv2
import pyrealsense2 as rs
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

img_height = 120
img_width = 160

PERCENT = 25


def resize_image(img):
    width = int(img.shape[1] * PERCENT / 100)
    height = int(img.shape[0] * PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


model_rgb = keras.Sequential(
    [
        layers.Conv2D(128, kernel_size=(3, 4), input_shape=(120, 160, 3), strides=(1, 1), padding='valid',
                      activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax'),
    ]
)

model_depth = keras.Sequential(
    [
        layers.Conv2D(128, kernel_size=(3, 4), input_shape=(120, 160, 3), strides=(1, 1), padding='valid',
                      activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax'),
    ]
)

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[0], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


def tester():
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions_rgb = []
    predictions_depth = []
    threshold = 0.7

    # Setup pipeline
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
    model_rgb.load_weights('rgb_only_new_weights.h5')
    model_depth.load_weights('depth_new_weights.h5')

    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)
        # Convert images to numpy arrays
        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        # preprocess the image
        rgb_image = resize_image(rgb_image)
        rgb_image = rgb_image.reshape((1, rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]))
        rgb_image = preprocess_input(rgb_image)

        # make the prediction

        res_rgb = model_rgb.predict(rgb_image)[0]
        predictions_rgb.append(np.argmax(res_rgb))

        # 3. Viz logic
        if np.unique(predictions_rgb[-10:])[0] == np.argmax(res_rgb):
            if res_rgb[np.argmax(res_rgb)] > threshold:
                print(actions[np.argmax(res_rgb)])
                if len(sentence) > 0:
                    if actions[np.argmax(res_rgb)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res_rgb)])
                else:
                    sentence.append(actions[np.argmax(res_rgb)])

        if len(sentence) > 5:
            sentence = sentence[-5:]
            print(sentence)

        # Viz probabilities
        # image = prob_viz(res_rgb, actions, rgb_image, colors)
        #
        # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        # cv2.putText(image, ' '.join(sentence), (3, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        #cv2.imshow('OpenCV Feed', rgb_image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    pipeline.stop()


CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tester()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
