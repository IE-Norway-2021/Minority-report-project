# Imports needed
import os

import cv2
from PIL import Image
import tensorflow as tf
from keras.engine.training_utils_v1 import unpack_validation_data
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

img_height = 120
img_width = 160
batch_size = 12
folder_name = 'rgb_image_dataset'
split_value = 0.1
EPOCHS = 20
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


def test2():
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test2()
