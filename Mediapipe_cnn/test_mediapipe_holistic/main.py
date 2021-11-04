# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mediapipe as mp
import math
import numpy as np
import cv2

mp_drawing = mp.drawing_utils
#mp_drawing_styles = mp.drawing_styles
mp_holistic = mp.solutions.hand

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def resize_and_show(im):
    h, w = im.shape[:2]
    if h < w:
        img = cv2.resize(im, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(im, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow(im)


with mp_holistic.Holistic(static_image_mode=True) as holistic:
    image = cv2.imread("tmp.jpg")
    name = "tmp.png"
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print the real-world 3D coordinates of nose in meters with the origin at
    # the center between hips.
    print('Nose world landmark:'),
    print(results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])

