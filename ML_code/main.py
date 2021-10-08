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
batch_size = 4
folder_name = 'rgb_160x120_image_dataset'
split_value = 0.8

initializer = tf.keras.initializers.HeNormal()


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def relu_u(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    # bn = BatchNormalization()(relu)
    return relu


def bottleneck(x: Tensor, kernels: int, dilation: int) -> Tensor:
    y = Conv2D(kernel_size=1,
               strides=1,
               filters=int(kernels / 4),
               padding="same", kernel_initializer=initializer)(x)
    y = relu_bn(y)

    y = Conv2D(kernel_size=(3, 4),
               strides=1,
               filters=int(kernels / 4),
               padding="same", kernel_initializer=initializer)(y)
    y = relu_bn(y)

    y = Conv2D(kernel_size=1,
               strides=1,
               filters=kernels,
               padding="same", kernel_initializer=initializer)(y)
    y = relu_bn(y)

    out = Add()([x, y])

    y1 = Conv2D(kernel_size=1,
                strides=1,
                filters=int(kernels / 4),
                padding="same", kernel_initializer=initializer)(out)
    y1 = relu_bn(y1)

    y1 = Conv2D(kernel_size=(3, 4),
                strides=1,
                filters=int(kernels / 4),
                dilation_rate=dilation,
                padding="same", kernel_initializer=initializer)(y1)
    y1 = relu_bn(y1)

    y1 = Conv2D(kernel_size=1,
                strides=1,
                filters=kernels,
                padding="same", kernel_initializer=initializer)(y1)
    y1 = relu_bn(y1)

    out1 = Add()([out, y1])

    return out1


def create_net():
    inputs = Input(shape=(img_height, img_width, 3))

    t = Conv2D(kernel_size=(3, 4),
               strides=2,
               filters=64,
               padding="valid", kernel_initializer=initializer)(inputs)
    t = relu_bn(t)

    t = Conv2D(kernel_size=(3, 4),
               strides=2,
               filters=128,
               padding="valid", kernel_initializer=initializer)(t)
    t = relu_bn(t)

    t = bottleneck(t, kernels=128, dilation=2)

    t = Conv2D(kernel_size=(3, 4),
               strides=2,
               filters=256,
               padding="valid", kernel_initializer=initializer)(t)
    t = relu_bn(t)

    t = bottleneck(t, kernels=256, dilation=4)

    t = Conv2D(kernel_size=(3, 4),
               strides=2,
               filters=128,
               padding="valid", kernel_initializer=initializer)(t)
    t = relu_bn(t)

    t = bottleneck(t, kernels=128, dilation=8)

    t = Conv2D(kernel_size=(3, 4),
               strides=2,
               filters=64,
               padding="valid", kernel_initializer=initializer)(t)
    t = relu_bn(t)

    t = bottleneck(t, kernels=64, dilation=4)

    t = Conv2D(kernel_size=(3, 3),
               strides=2,
               filters=16,
               padding="same", kernel_initializer=initializer)(t)
    t = relu_bn(t)

    '''
    t=bottleneck(t,kernels=32,dilation=2)

    t = Conv2D(kernel_size=3,
               strides=2,
               filters=16,
               padding="valid")(t)
    t = relu_bn(t)
'''
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Old model usage, redundant
# model = keras.Sequential(
#     [
#         layers.Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
#                       input_shape=(img_width, img_height, 1)),
#         layers.Conv2D(16, 3, padding="same"),
#         layers.Conv2D(32, 3, padding="same"),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(100, input_shape=(784,), activation='relu'),
#         layers.Dense(10, activation='softmax'),
#     ]
# )


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
    net = create_net()

    net.fit(ds_train, epochs=50, verbose=1, validation_data=ds_validation)
    # evaluate model
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    y_ = net.predict(ds_train)
    y_pred = np.argmax(tf.nn.softmax(y_), axis=1)

    con_mat = tf.math.confusion_matrix(labels=ds_train, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(ds_train, y_pred))
    from sklearn.metrics import accuracy_score
    accuracy_score(ds_train, y_pred)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
