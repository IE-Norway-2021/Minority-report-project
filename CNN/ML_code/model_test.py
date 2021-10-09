import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_height = 120
img_width = 160
batch_size = 12
folder_name = 'rgb_160x120_image_dataset'
split_value = 0.1


def tmp():
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(img_width, img_height, 3)),
            layers.Conv2D(filters=128, kernel_size=(4, 5), strides=(1, 1), padding='valid'),
            layers.Activation(activation="relu"),
            layers.Conv2D(filters=64, kernel_size=(2, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(),
            layers.Dropout(0.25),
            layers.Conv2D(filters=256, kernel_size=(2, 3), padding="same", activation="relu"),
            layers.Conv2D(filters=128, kernel_size=(1, 2), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=2, padding="same", activation="relu"),
            layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(),
            layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.8),
            layers.Dense(512, activation='relu'),
            layers.Dense(200),
            layers.Dropout(0.8),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()


if __name__ == '__main__':
    tmp()
