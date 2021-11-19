"""Landmark_drawer
This script allows a user to write on existing photos the landmarks for both hands using the mediapipe library

To use, just make sure the DEST_PATH and OLD_PATH variables are correct for your use case
"""
import mediapipe as mp
import os

import numpy as np
from PIL import Image
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DEST_PATH = 'rgb_landmark_image_dataset'
OLD_PATH = 'rgb_image_dataset'
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def landmark_drawer(imagePath, path):
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        image = cv2.imread(imagePath)
        results = holistic.process(image)
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        cv2.imwrite(path, image)


def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']', f' {percent:.0f}%', sep='', end='', flush=True)


def main():
    dir_counter = -1
    for dirpath, dirnames, files in os.walk(OLD_PATH):
        print(f'\nProcessing for dir {dir_counter}...\n')
        counter = 0
        for file_name in files:
            percent = int((counter * 100) / len(files))
            image = dirpath + '\\' + file_name
            landmark_drawer(image, DEST_PATH + dirpath.replace(OLD_PATH, '') + '\\' + file_name)
            counter += 1
            progress(percent)
        dir_counter += 1


if __name__ == '__main__':
    main()
