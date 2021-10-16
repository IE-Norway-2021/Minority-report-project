import cv2
import pyrealsense2 as rs
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

img_height = 120
img_width = 160

model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(img_width, img_height, 3)),
        layers.Conv2D(25, kernel_size=(3, 4), strides=(1, 1), padding='valid', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
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
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


def tester():
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

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
    model.load_weights('rgb_only_weights.h5')

    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 0)
        # Convert images to numpy arrays
        image = np.asanyarray(color_frame.get_data())
        print(image.shape)
        # preprocess the image
        my_image = img_to_array(color_frame.get_data())
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        my_image = preprocess_input(my_image)

        # make the prediction
        res = model.predict(my_image)[0]
        print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))

        # 3. Viz logic
        if np.unique(predictions[-10:])[0] == np.argmax(res):
            if res[np.argmax(res)] > threshold:

                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        # Viz probabilities
        image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

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
