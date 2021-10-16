import os
# importing Image class from PIL package
from PIL import Image
import cv2

OLD_PATH = 'Dataset/depth'
DEST_PATH = OLD_PATH
PERCENT = 25
WIDTH = 160  # defines ratio
HEIGHT = 120


def redim_image(image, new_path):
    img = cv2.imread(image)
    width = int(img.shape[1] * PERCENT / 100)
    height = int(img.shape[0] * PERCENT / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_path, resized)


def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']', f' {percent:.0f}%', sep='', end='', flush=True)


def redim_image_in_file():
    for dirpath, dirnames, files in os.walk(OLD_PATH):
        counter = 0
        for file_name in files:
            percent = int((counter * 100) / len(files))
            image = dirpath + '\\' + file_name
            redim_image(image, DEST_PATH + dirpath.replace(OLD_PATH, '') + '\\' + file_name)
            counter += 1
            progress(percent)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    redim_image_in_file()
