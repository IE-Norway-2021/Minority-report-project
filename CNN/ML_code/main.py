# Imports needed
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, \
    Dense
from tensorflow.python.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 120
img_width = 160
batch_size = 12
folder_name = 'rgb_160x120_image_dataset'
split_value = 0.1

# Old model usage, redundant

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
        layers.Dense(100, activation='relu'),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(ds_train, epochs=50, verbose=1, validation_data=ds_validation)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
