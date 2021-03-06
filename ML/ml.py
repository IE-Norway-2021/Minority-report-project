"""
This script contains the main code for training models using our dataset, and also functions that allow to create graphs from the results

"""
# Imports needed
import os
import gc

import cv2
import tensorflow as tf
from enum import Enum
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
from tensorflow.keras.utils import plot_model
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



img_height = 120
img_width = 160
batch_size = 4
folder_name = 'video_test_dataset'
split_value = 0.1
EPOCHS = 80
num_of_folds = 10
INIT_LR = 0.00001
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
PERCENT = 25

# PHOTO part
# Below are functions for training the models based on the image dataset. The "test" function is useful only for training, but does not allow to use 
# the trained models. The function rgb_new and depth_new use a new method for loading images that allows using the trained model in real time scenarios

def resize_image(img):
    width = int(img.shape[1] * PERCENT / 100)
    height = int(img.shape[0] * PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized



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

    model = keras.Sequential(
        [
            layers.Conv2D(128, kernel_size=(3, 4), input_shape=(img_width, img_height, 3), strides=(1, 1),
                          padding='valid',
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


# Video part

movements = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out'])
sequence_length = 40

# This class is used to make sure we clean all data beetween epochs to avoid memory issues
class RemoveGarbageCallback(tf.keras.callbacks.Callback):
    @staticmethod
    def on_epoch_end(epoch, logs=None):
        gc.collect()

# This enum defines the different types of models used. 
# the models marked as beginning, middle or end are based on an experiment we did that would cut the sequence in 3 parts and do predictions 
# on each and combine all results to check for gestures. Did not work
class Dataset_type(Enum):
    Normal = 0 # A normal dataset, with reduced parameters compared to the model used in the article. this was our base model
    reduced_4 = 1 # a model based on normal but only using 1 of every 4 frames
    reduced_4_beginning = 2 
    reduced_4_middle = 3
    reduced_4_end = 4
    reduced_2_beginning = 5
    reduced_2_middle = 6
    reduced_2_end = 7
    full_beginning = 8
    reduced_2 = 9 # a model based on normal but only using 1 of every 2 frames
    default_full_2_4 = 10 # the model used in the article. the inputs in the getModel function allow to 
                          # define for what type (fill, 1 of 2 or 1 of 4) of dataset it is going to be used
    reduced_2_pi = 11 # the model used in pi. based on default_full_2_4 but with less layers


# This function will return a model based on the dataset type given. for default_full_2_4 the input shape input needs to be given
def getModel(dataset_type, input_shape=(0, 0, 0, 0)):
    if dataset_type is Dataset_type.Normal:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(8, kernel_size=(3, 3, 4), input_shape=(40, 120, 160, 3), strides=(1, 1, 1),
                              padding='valid', activation='relu'),
                layers.MaxPool3D(),
                layers.Conv3D(16, 3, padding="same", activation="relu"),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
                layers.Conv3D(8, 3, padding="same", activation="relu"),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(30, activation='relu'),
                layers.Dense(15, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    elif dataset_type is Dataset_type.default_full_2_4:  # model used in the article
        vid_input = Input(input_shape)
        x = Conv3D(16, kernel_size=(3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu')(vid_input)
        x = MaxPooling3D(padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv3D(32, kernel_size=(3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu')(x)
        x = MaxPooling3D(padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv3D(16, kernel_size=(3, 3, 4), strides=(1, 1, 1), padding='same', activation='relu')(x)
        x = MaxPooling3D(padding="same")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(60, activation='relu')(x)
        x = Dense(30, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(6, activation='softmax')(x)
        model_vid = Model(vid_input, x, name='Custom_CNN')
        return model_vid
    elif dataset_type is Dataset_type.reduced_2_pi:  # model used for faster inference on the raspberry pi
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=input_shape, strides=(1, 1, 1),
                              padding='valid', activation='relu'),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
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
    elif dataset_type is Dataset_type.reduced_4:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(10, 120, 160, 3), strides=(1, 1, 1),
                              padding='valid', activation='relu'),
                layers.MaxPool3D(),
                layers.Conv3D(32, 3, padding="same", activation="relu"),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
                layers.Conv3D(16, 3, padding="same", activation="relu"),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(120, activation='relu'),
                layers.Dense(60, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    elif dataset_type is Dataset_type.reduced_2:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(20, 120, 160, 3), strides=(1, 1, 1),
                              padding='valid', activation='relu'),
                layers.MaxPool3D(),
                layers.Conv3D(32, 3, padding="same", activation="relu"),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
                layers.Conv3D(16, 3, padding="same", activation="relu"),
                layers.MaxPool3D(),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(120, activation='relu'),
                layers.Dense(60, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    elif dataset_type is Dataset_type.reduced_4_beginning or dataset_type is Dataset_type.reduced_4_end:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(3, 120, 160, 3), strides=(1, 1, 1),
                              padding='same', activation='relu'),
                layers.MaxPool3D(padding="same"),
                layers.Conv3D(32, 1, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Conv3D(16, 1, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(120, activation='relu'),
                layers.Dense(60, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    elif dataset_type is Dataset_type.reduced_4_middle:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(4, 120, 160, 3), strides=(1, 1, 1),
                              padding='same', activation='relu'),
                layers.MaxPool3D(padding="same"),
                layers.Conv3D(32, 3, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Conv3D(16, 3, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(120, activation='relu'),
                layers.Dense(60, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    elif dataset_type is Dataset_type.reduced_2_beginning or dataset_type is Dataset_type.reduced_2_end \
            or dataset_type is Dataset_type.full_beginning:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(6, 120, 160, 3), strides=(1, 1, 1),
                              padding='same', activation='relu'),
                layers.MaxPool3D(padding="same"),
                layers.Conv3D(32, 1, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Conv3D(16, 1, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(120, activation='relu'),
                layers.Dense(60, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    elif dataset_type is Dataset_type.reduced_2_middle:
        model_vid = keras.Sequential(
            [
                layers.Conv3D(16, kernel_size=(3, 3, 4), input_shape=(8, 120, 160, 3), strides=(1, 1, 1),
                              padding='same', activation='relu'),
                layers.MaxPool3D(padding="same"),
                layers.Conv3D(32, 3, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Conv3D(16, 3, padding="same", activation="relu"),
                layers.MaxPool3D(padding="same"),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(120, activation='relu'),
                layers.Dense(60, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(6, activation='softmax'),
            ]
        )
    return model_vid



# this is the main training function. It will load all required sequences from root, get a model based on dataset_type,
# and either train the model or do a kfold training. We can also only generate a confusion matrix, it will in that case 
# load weights of the model with the given name (assuming it has already been trained)
def video_ml(root, name, dataset_type=Dataset_type.Normal, input_shape=(0, 0, 0, 0), kfold=False,
             only_confusion_matrix=False):
    print('Starting Image loading...')
    label_map = {label: num for num, label in enumerate(movements)}
    sequences, labels = [], []
    for movement in movements:
        for dirpath, dirnames, files in os.walk(os.path.join(root, movement)):
            sequence = []
            if len(files) != 0:
                if dataset_type is Dataset_type.Normal:
                    for i in range(sequence_length):
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.reduced_4:
                    for i in range(sequence_length):
                        if i % 4 == 0:
                            img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                            sequence.append(img)
                elif dataset_type is Dataset_type.reduced_2:
                    for i in range(sequence_length):
                        if i % 2 == 0:
                            img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                            sequence.append(img)
                elif dataset_type is Dataset_type.default_full_2_4 or dataset_type is Dataset_type.reduced_2_pi:
                    for i in range(sequence_length):
                        if i % (sequence_length / input_shape[0]) == 0:
                            img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                            sequence.append(img)
                elif dataset_type is Dataset_type.reduced_4_beginning:
                    for i in [0, 4, 8]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.reduced_4_middle:
                    for i in [12, 16, 20, 24]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.reduced_4_end:
                    for i in [28, 32, 36]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.reduced_2_beginning:
                    for i in [0, 2, 4, 6, 8, 10]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.reduced_2_middle:
                    for i in [12, 14, 16, 18, 20, 22, 24, 26]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.reduced_2_end:
                    for i in [28, 30, 32, 34, 36, 38]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
                elif dataset_type is Dataset_type.full_beginning:
                    for i in [0, 1, 2, 3, 4, 5]:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
            if len(sequence) > 0:
                sequences.append(sequence)
                labels.append(label_map[movement])
    print('Image loading done! Choosing model and starting train set creation...')
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    del labels
    del sequences
    gc.collect()

    if kfold:
        test_acc_per_fold = []
        train_acc_per_fold = []
        epochs = 30
        fold_no = 0
        kfold = KFold(n_splits=num_of_folds, shuffle=True)
        for train, test in kfold.split(X, y):
            model_vid = getModel(dataset_type, input_shape)
            model_vid.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                              metrics=["accuracy"],
                              )
            history = model_vid.fit(X[train], y[train], epochs=epochs, verbose=0, validation_data=(X[test], y[test]))
            test_loss, test_acc = model_vid.evaluate(X[test], y[test])
            train_acc = history.history['accuracy'][epochs - 1]
            print("test accuracy in fold {} : {} %".format(fold_no + 1, test_acc * 100))
            print("train accuracy in fold {} : {} %".format(fold_no + 1, train_acc * 100))
            fold_no = fold_no + 1
            test_acc_per_fold.append(test_acc * 100)
            train_acc_per_fold.append(train_acc * 100)
            tf.keras.backend.clear_session()  # clear memory beetween folds
            gc.collect()
        print(f'> Overall Test Accuracy for {name}: {np.mean(test_acc_per_fold)} (+- {np.std(test_acc_per_fold)})')
        print(f'> Overall Train Accuracy for {name}: {np.mean(train_acc_per_fold)} (+- {np.std(train_acc_per_fold)})')
        # save arrays
        np.save(f'output/kfold_{name}_test_acc.npy', test_acc_per_fold)
        np.save(f'output/kfold_{name}_train_acc.npy', train_acc_per_fold)
    else:
        model_vid = getModel(dataset_type, input_shape)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_value)
        print('Train set creation done!')
        # erase no longer used variables
        del X
        del y
        gc.collect()

        if not only_confusion_matrix:
            model_vid.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
                loss='categorical_crossentropy', metrics=["accuracy"],
            )
            model_vid.summary()
            history = model_vid.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val),
                                    callbacks=[RemoveGarbageCallback()])

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
            plt.savefig(f'output/{name}_accuracy_results.png')
            plt.clf()

            plt.figure(figsize=(15, 15))
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.savefig(f'output/{name}_loss_results.png')
            plt.clf()
            with open(f'output/{name}_training_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            model_vid.save(f'output/{name}_weights.h5')
        else:
            model_vid.load_weights(f'output/{name}_weights.h5')
        # confusion matrix
        generateConfusionMatrix(model_vid, X_val, y_val, name)

        del X_val
        del X_train
        del y_val
        del y_train
        gc.collect()

# The following functions allow to train multiple models in a single function

def train_normal():
    print('Doing rgb training...')
    video_ml('video_dataset/rgb', 'video_rgb')
    print('Doing depth training...')
    video_ml('video_dataset/depth', 'video_depth')


def train_reduced_4():
    print('Doing rgb reduced_4 training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4', dataset_type=Dataset_type.reduced_4)
    print('Doing depth reduced_4 training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4', dataset_type=Dataset_type.reduced_4)


def train_reduced_4_beg_mid_end():
    print('Doing rgb reduced_4 beginning training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4_beginning', dataset_type=Dataset_type.reduced_4_beginning)
    print('Doing depth reduced_4 beginning training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4_beginning', dataset_type=Dataset_type.reduced_4_beginning)
    print('Doing rgb reduced_4 middle training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4_middle', dataset_type=Dataset_type.reduced_4_middle)
    print('Doing depth reduced_4 middle training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4_middle', dataset_type=Dataset_type.reduced_4_middle)
    print('Doing rgb reduced_4 end training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4_end', dataset_type=Dataset_type.reduced_4_end)
    print('Doing depth reduced_4 end training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4_end', dataset_type=Dataset_type.reduced_4_end)


def train_reduced_2_beg_mid_end():
    print('Doing rgb reduced_2 beginning training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2_beginning', dataset_type=Dataset_type.reduced_2_beginning)
    print('Doing depth reduced_2 beginning training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2_beginning', dataset_type=Dataset_type.reduced_2_beginning)
    print('Doing rgb reduced_2 middle training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2_middle', dataset_type=Dataset_type.reduced_2_middle)
    print('Doing depth reduced_2 middle training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2_middle', dataset_type=Dataset_type.reduced_2_middle)
    print('Doing rgb reduced_2 end training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2_end', dataset_type=Dataset_type.reduced_2_end)
    print('Doing depth reduced_2 end training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2_end', dataset_type=Dataset_type.reduced_2_end)


def train_reduced_2():
    print('Doing rgb reduced_2 training...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2', dataset_type=Dataset_type.reduced_2)
    print('Doing depth reduced_2 training...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2', dataset_type=Dataset_type.reduced_2)


def kfold_for_reduced_2_4_and_full():
    print('Doing rgb reduced_4 kfold...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4', kfold=True, dataset_type=Dataset_type.default_full_2_4,
             input_shape=(10, 120, 160, 3))
    print('Doing depth reduced_4 kfold...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4', kfold=True, dataset_type=Dataset_type.default_full_2_4,
             input_shape=(10, 120, 160, 3))
    print('Doing rgb reduced_2 kfold...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2', kfold=True, dataset_type=Dataset_type.default_full_2_4,
             input_shape=(20, 120, 160, 3))
    print('Doing depth reduced_2 kfold...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2', kfold=True, dataset_type=Dataset_type.default_full_2_4,
             input_shape=(20, 120, 160, 3))
    print('Doing rgb full kfold...')
    video_ml('video_dataset/rgb', 'video_rgb_full', kfold=True, dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3))
    print('Doing depth full kfold...')
    video_ml('video_dataset/depth', 'video_depth_full', kfold=True, dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3))


def train_full_with_full_model():
    print('Doing rgb full with full model...')
    video_ml('video_dataset/rgb', 'video_rgb_full_heavy_model', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3))
    print('Doing depth full with full model...')
    video_ml('video_dataset/depth', 'video_depth_full_heavy_model', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3))


def train_reduced_2_pi():
    print('Doing rgb reduced 2 pi...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2_pi', dataset_type=Dataset_type.reduced_2_pi,
             input_shape=(20, 120, 160, 3))
    print('Doing depth reduced 2 pi...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2_pi', dataset_type=Dataset_type.reduced_2_pi,
             input_shape=(20, 120, 160, 3))


def train_with_main_model():
    print('Doing rgb reduced_2...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(20, 120, 160, 3))
    print('Doing depth reduced_2 ...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(20, 120, 160, 3))
    print('Doing rgb reduced_4 ...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(10, 120, 160, 3))
    print('Doing depth reduced_4 ...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(10, 120, 160, 3))
    print('Doing rgb full ...')
    video_ml('video_dataset/rgb', 'video_rgb_full', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3))
    print('Doing depth full ...')
    video_ml('video_dataset/depth', 'video_depth_full', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3))


def generate_main_confusion_matrices():
    print('Doing rgb reduced_2...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_2', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(20, 120, 160, 3), only_confusion_matrix=True)
    print('Doing depth reduced_2 ...')
    video_ml('video_dataset/depth', 'video_depth_reduced_2', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(20, 120, 160, 3), only_confusion_matrix=True)
    print('Doing rgb reduced_4 ...')
    video_ml('video_dataset/rgb', 'video_rgb_reduced_4', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(10, 120, 160, 3), only_confusion_matrix=True)
    print('Doing depth reduced_4 ...')
    video_ml('video_dataset/depth', 'video_depth_reduced_4', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(10, 120, 160, 3), only_confusion_matrix=True)
    print('Doing rgb full ...')
    video_ml('video_dataset/rgb', 'video_rgb_full', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3), only_confusion_matrix=True)
    print('Doing depth full ...')
    video_ml('video_dataset/depth', 'video_depth_full', dataset_type=Dataset_type.default_full_2_4,
             input_shape=(40, 120, 160, 3), only_confusion_matrix=True)


# This function will only plot the model using the keras function
def generate_model_plot(dataset_type, name, input_shape=(0, 0, 0, 0)):
    model = getModel(dataset_type, input_shape=input_shape)
    plot_model(model, to_file=f'output/{name}.png')

# This function will generate a confusion matrix based on the given parameters. Used in the video_ml function
def generateConfusionMatrix(model_vid, X_val, y_val, name):
    Y_te = np.array(tf.math.argmax(model_vid.predict(X_val), 1))
    y_val = np.array(tf.math.argmax(y_val, 1))
    acc = metrics.accuracy_score(y_val, Y_te)
    print("test accuracy =", acc * 100, "%\n")
    print(classification_report(y_val, Y_te))
    con_mat = tf.math.confusion_matrix(labels=y_val, predictions=Y_te).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=movements, columns=movements)
    plt.figure()
    sns.heatmap(con_mat_df, annot=True, cmap="RdPu")
    plt.title('Convolution Neural Newtork')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'output/{name}_confusion_matrix.png', bbox_inches='tight')
    plt.clf()

## These two function bellow are used to produce information for the article. 
## Before using them please use this function train_with_main_model to generate results and do kfold training aswell 

# This function generates a graph with the kfold results of the training. It assumes the results are already present
def generate_kfold_results_graph():
    data_train = [[], [], [], [], [], [], [], [], [], []]
    data_test = [[], [], [], [], [], [], [], [], [], []]
    path = 'ML_video_results/Kfold'
    for result_name in ['kfold_video_rgb_full_train_acc.npy', 'kfold_video_depth_full_train_acc.npy',
                        'kfold_video_rgb_reduced_2_train_acc.npy', 'kfold_video_depth_reduced_2_train_acc.npy',
                        'kfold_video_rgb_reduced_4_train_acc.npy', 'kfold_video_depth_reduced_4_train_acc.npy']:
        result = np.load(f'{path}/{result_name}')
        for i in range(len(result)):
            data_train[i].append(result[i])
    for result_name in ['kfold_video_rgb_full_test_acc.npy', 'kfold_video_depth_full_test_acc.npy',
                        'kfold_video_rgb_reduced_2_test_acc.npy', 'kfold_video_depth_reduced_2_test_acc.npy',
                        'kfold_video_rgb_reduced_4_test_acc.npy', 'kfold_video_depth_reduced_4_test_acc.npy', ]:
        result = np.load(f'{path}/{result_name}')
        for i in range(len(result)):
            data_test[i].append(result[i])
    for data, name in [(data_train, "training"), (data_test, "testing")]:
        plotdata = pd.DataFrame({"fold 1": data[0], "fold 2": data[1], "fold 3": data[2],
                                 "fold 4": data[3], "fold 5": data[4],
                                 "fold 6": data[5], "fold 7": data[6], "fold 8": data[7],
                                 "fold 9": data[8], "fold 10": data[9]},
                                index=["rgb_full", "depth_full", "rgb_reduced_2", "depth_reduced_2", "rgb_reduced_4",
                                       "depth_reduced_4"])
        sns.set_style("dark")
        plotdata.plot(kind="bar", figsize=(10, 6)).legend(loc='upper right', ncol=5, bbox_to_anchor=(1, 1.2))
        plt.title(f'Kfold results for {name}')
        plt.ylim([85, 100])
        plt.xlabel("Dataset types")
        plt.ylabel("Accuracy")
        plt.savefig(f'{path}/kfold_{name}_graph.png', bbox_inches='tight')


# This function generates the accuracy and loss graphs using the results of training. 
# It assumes the results exists
def generateAccuracyLossGraphs():
    data_rgb = []
    data_depth = []
    path = 'ML_video_results/Training'
    for result_name in ['video_rgb_full_training_history', 'video_rgb_reduced_2_training_history',
                        'video_rgb_reduced_4_training_history']:
        result = pickle.load(open(f'{path}/{result_name}', "rb"))
        data_rgb.append(result)
    for result_name in ['video_depth_full_training_history', 'video_depth_reduced_2_training_history',
                        'video_depth_reduced_4_training_history']:
        result = pickle.load(open(f'{path}/{result_name}', "rb"))
        data_depth.append(result)
    for type, val_type, name in [('accuracy', 'val_accuracy', 'Accuracy'), ('loss', 'val_loss', 'Loss')]:
        for stream_name, data in [('rgb', data_rgb), ('depth', data_depth)]:
            fontsize = 26
            plotdata = pd.DataFrame(
                {f'{name} for proposed model/full': data[0][type],
                 f'Validation {name} for proposed model/full': data[0][val_type],
                 f'{name} for proposed model/1 of 2': data[1][type],
                 f'Validation {name} for proposed model/1 of 2': data[1][val_type],
                 f'{name} for proposed model/1 of 4': data[2][type],
                 f'Validation {name} for proposed model/1 of 4': data[2][val_type]})
            plotdata.plot(kind="line", figsize=(25, 13), lw=3, fontsize=fontsize - 3).legend(loc='upper right', ncol=2,
                                                                                             bbox_to_anchor=(
                                                                                                 0.97, 1.26),
                                                                                             fontsize=fontsize)
            plt.title(f'{name} and Validation {name} during training for {stream_name}',
                      fontdict={'fontsize': fontsize})
            plt.xlabel("Epochs", fontdict={'fontsize': fontsize})
            plt.ylabel(name, fontdict={'fontsize': fontsize})
            plt.savefig(f'{path}/training_{type}_{stream_name}_graph.png', bbox_inches='tight')


if __name__ == '__main__':
    # uncomment here to improve performance on supported gpus
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    os.makedirs("output", exist_ok=True)
    model = getModel(Dataset_type.default_full_2_4, (10, 120, 160, 3))
    model.summary()
