"""Landmark_drawer
This script allows a user to write on existing video the landmarks for both hands using the mediapipe library

To use, just make sure the DEST_PATH and OLD_PATH variables are correct for your use case
"""
import threading

import mediapipe as mp
import os
import time

import concurrent.futures
import cv2
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DEST_PATH = 'video_landmark_dataset/rgb'
OLD_PATH = 'tmp/rgb'
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def landmark_extractor(imagePath):
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        image = cv2.imread(imagePath)
        results = holistic.process(image)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, lh, rh])


def progress(percent_dir=0, width=30):
    left = width * percent_dir // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']', f' {percent_dir:.0f}%', sep='', end='', flush=True)


movements = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out'])


def main():
    percent = 0
    for movement in movements:
        dir_counter = 0
        length = 1
        print(f'\nDoing movement : {movement}\n')
        for dirpath, dirnames, files in os.walk(os.path.join(OLD_PATH, movement)):
            if len(files) != 0:
                percent = int((dir_counter * 100) / length)
                os.makedirs(DEST_PATH + dirpath.replace(OLD_PATH, ''))
            else:
                length = len(dirnames)
            for file_name in files:
                image = dirpath + '\\' + file_name
                landmarks = landmark_extractor(image)
                np.save(DEST_PATH + dirpath.replace(OLD_PATH, '') + '\\' + file_name.replace('.png', ''), landmarks)
                progress(percent)
            if len(files) != 0:
                dir_counter += 1


running = [1]
nbr_of_threads = 4

shared_array = []

wait_lock = threading.Semaphore(0)
mutex = threading.Semaphore(1)


def img_processing(name):
    print(f'Thread {name} starting')
    while 1:
        if len(shared_array) == 0:
            wait_lock.acquire()
        if running[0] == 0:
            break
        mutex.acquire()
        file_name = shared_array[0][0]
        dirpath = shared_array[0][1]
        shared_array.pop(0)
        mutex.release()
        image = dirpath + '\\' + file_name
        landmarks = landmark_extractor(image)
        np.save(DEST_PATH + dirpath.replace(OLD_PATH, '') + '\\' + file_name.replace('.png', ''), landmarks)


def main_threaded():
    print('Creating folders...')
    for movement in movements:
        print(f'\rDoing movement : {movement}')
        for dirpath, dirnames, files in os.walk(os.path.join(OLD_PATH, movement)):
            os.makedirs(DEST_PATH + dirpath.replace(OLD_PATH, ''), exist_ok=True)

    print('Starting threads for npy array processing and saving...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=nbr_of_threads) as executor:
        executor.map(img_processing, range(nbr_of_threads))
        time.sleep(4)
        for movement in movements:
            dir_counter = 0
            length = 1
            print(f'Doing movement : {movement}', flush=True)
            for dirpath, dirnames, files in os.walk(os.path.join(OLD_PATH, movement)):
                while True:
                    mutex.acquire()
                    if len(shared_array) <= nbr_of_threads:
                        mutex.release()
                        break
                    mutex.release()
                    time.sleep(0.5)
                if len(files) == 0:
                    length = len(dirnames)
                for file_name in files:
                    mutex.acquire()
                    shared_array.append([file_name, dirpath])
                    mutex.release()
                    wait_lock.release()
                if len(files) != 0:
                    dir_counter += 1
                    percent = int((dir_counter * 100) / length)
                    progress(percent)
        while True:
            mutex.acquire()
            if len(shared_array) == 0:
                mutex.release()
                break
            mutex.release()
            time.sleep(0.5)
        running[0] = 0
        for i in range(nbr_of_threads):
            wait_lock.release()


if __name__ == '__main__':
    start_time = time.time()
    main_threaded()
    print("--- %s seconds ---" % (time.time() - start_time))
