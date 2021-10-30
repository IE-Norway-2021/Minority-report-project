# Imports needed
import os

import cv2
import sys
from PIL import Image
import tensorflow as tf
from keras.engine.training_utils_v1 import unpack_validation_data
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

img_height = 120
img_width = 160
batch_size = 4
folder_name = 'video_test_dataset'
split_value = 0.1
EPOCHS = 300
INIT_LR = 0.00001
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
PERCENT = 25


def resize_image(img):
    width = int(img.shape[1] * PERCENT / 100)
    height = int(img.shape[0] * PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


model = keras.Sequential(
    [
        layers.Conv2D(128, kernel_size=(3, 4), input_shape=(img_width, img_height, 3), strides=(1, 1), padding='valid',
                      activation='relu'),
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


#                      METHOD 1
# ==================================================== #
#             Using dataset_from_directory             #
# ==================================================== #
def test():
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        folder_name,
        labels="inferred",
        label_mode="int",  # categorical, binary
        class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_width, img_height),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=split_value,
        subset="training",
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        folder_name,
        labels="inferred",
        label_mode="int",  # categorical, binary
        class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_width, img_height),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=split_value,
        subset="validation",
    )
    # ds_train = ds_train.map(augment)
    # Custom Loops
    # for epochs in range(10):
    #     for x, y in ds_train:
    #         # train here
    #         pass
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(ds_train, epochs=EPOCHS, verbose=1, validation_data=ds_validation)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    model.save('rgb_landmark_weights.h5')


def rgb_new():
    print('Starting Image loading...')
    label_map = {label: num for num, label in enumerate(actions)}
    images, labels = [], []
    for action in actions:
        for dirpath, dirnames, files in os.walk(os.path.join(folder_name, action)):
            for file_name in files:
                img = cv2.imread(os.path.join(folder_name, action, file_name))
                images.append(resize_image(img))
                labels.append(label_map[action])
    print('Image loading done! Starting train set creation...')
    X = np.array(images)
    y = to_categorical(labels).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_value)
    print('Train set creation done!')

    model2 = keras.Sequential(
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

    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
        loss='categorical_crossentropy', metrics=["accuracy"],
    )
    model2.summary()
    history = model2.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    model2.save('rgb_only_new_weights.h5')


def depth_new():
    label_map = {label: num for num, label in enumerate(actions)}
    images, labels = [], []
    for action in actions:
        for dirpath, dirnames, files in os.walk(os.path.join(folder_name, action)):
            action_set = []
            for file_name in files:
                img = cv2.imread(os.path.join(folder_name, action, file_name))
                if np.array(img).shape != (480, 640, 3):
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                if np.array(img).shape != (480, 640, 3):
                    print('Error')
                images.append(resize_image(img))
                labels.append(label_map[action])
    X = np.array(images)
    y = to_categorical(labels).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_value)
    model3 = keras.Sequential(
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

    model3.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
        loss='categorical_crossentropy', metrics=["accuracy"],
    )
    model3.summary()
    history = model3.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    model3.save('depth_new_weights.h5')


movements = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out'])
sequence_length = 40


def video_ml(root, name, reduced=False):
    print('Starting Image loading...')
    label_map = {label: num for num, label in enumerate(movements)}
    sequences, labels = [], []
    for movement in movements:
        for dirpath, dirnames, files in os.walk(os.path.join(root, movement)):
            sequence = []
            if len(files) != 0:
                if reduced:
                    for i in range(sequence_length):
                        if i % 4 == 0:
                            img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                            sequence.append(img)
                else:
                    for i in range(sequence_length):
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
            if len(sequence) > 0:
                sequences.append(sequence)
                labels.append(label_map[movement])
    print('Image loading done! Starting train set creation...')
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_value)
    print('Train set creation done!')

    if reduced:
        model_vid = keras.Sequential(
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
                layers.Dense(100, activation='relu'),
                layers.Dense(50, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    else:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(8, kernel_size=(3, 3, 4), input_shape=(40, 120, 160, 3), strides=(1, 1, 1),
                              padding='valid',
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

    model_vid.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
        loss='categorical_crossentropy', metrics=["accuracy"],
    )
    model_vid.summary()
    history = model_vid.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(15, 15))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig(f'{name}_accuracy_results.png')

    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f'{name}_loss_results.png')
    np.save(f'{name}_training_history.npy', history)
    model_vid.save(f'{name}_weights.h5')
    yhat = model_vid.predict(X_val)
    ytrue = np.argmax(y_val, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print(multilabel_confusion_matrix(ytrue, yhat))
    np.save(f'{name}_confusion_matrix.npy', multilabel_confusion_matrix(ytrue, yhat))


def video_rgb_ml():
    video_ml('video_dataset/rgb', 'video_rgb')


def video_depth_ml():
    video_ml('video_dataset/depth', 'video_depth')


def video_rgb_reduced_ml():
    video_ml('video_dataset/rgb', 'video_rgb_reduced', reduced=True)


def video_depth_reduced_ml():
    video_ml('video_dataset/depth', 'video_depth_reduced', reduced=True)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Redirect output
    sys.stdout = Logger()
    print('Doing rgb reduced training...')
    video_rgb_reduced_ml()
    print('Doing rgb training...')
    video_rgb_ml()
    print('Doing depth training...')
    video_depth_ml()
    print('Doing depth reduced training...')
    video_depth_reduced_ml()
