"""redim_image
Redimensionates all images to 25% for all images in the selected directory and saves them in a new directory

"""
import os
# importing Image class from PIL package
from PIL import Image

DEST_PATH = 'redim_dossier'
OLD_PATH = 'dossier'
WIDTH = 160  # defines ratio
HEIGHT = 120


def redim_image(image, new_path):
    image = Image.open(image)
    SIZE = (WIDTH, HEIGHT)
    # respects ratio
    image.thumbnail(SIZE)
    # creating thumbnail
    image.save(new_path)


def redim_image_in_file():
    for dirpath, dirnames, files in os.walk(OLD_PATH):
        for file_name in files:
            image = dirpath + '\\' + file_name
            redim_image(image, DEST_PATH + dirpath.replace(OLD_PATH, '') + '\\' + file_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    redim_image_in_file()
