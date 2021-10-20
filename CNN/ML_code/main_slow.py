# Imports needed
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import seaborn as sns

img_height = 120
img_width = 160
batch_size = 12
folder_name = 'rgb_160x120_image_dataset'
split_value = 0.1
EPOCHS = 50
INIT_LR = 0.00001

# Old model usage, redundant
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

    epochs_range = range(100)

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
