# Imports needed
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 480
img_width = 1488
batch_size = 128
folder_name = 'tbd'

model = keras.Sequential(
    [
        layers.Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                      input_shape=(img_width, img_height, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(100, input_shape=(784,), activation='relu'),
        layers.Dense(10, activation='softmax'),
    ]
)


def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


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
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="training",
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        folder_name,
        labels="inferred",
        label_mode="int",  # categorical, binary
        class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="validation",
    )
    ds_train = ds_train.map(augment)
    # Custom Loops
    for epochs in range(10):
        for x, y in ds_train:
            # train here
            pass
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=["accuracy"],
    )
    model.fit(ds_train, epochs=10, verbose=2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
