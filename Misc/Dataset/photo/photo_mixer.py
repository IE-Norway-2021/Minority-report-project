"""photo mixer
A script that takes the images in depth and rgb of a same shot and combines them into a single png file. The depth and
rgb images need to be all in the same directory
"""
import os

import matplotlib.image as mpimg
import numpy as np  # fundamental package for scientific computing

dest_dir_path = 'dest_dir\\'


def rgbd_function(imageRgb, imageDepth, imagePath):
    color = np.asanyarray(imageRgb)
    colorized_depth = np.asanyarray(imageDepth)
    # Show the two frames together:
    images = np.hstack((color, colorized_depth))
    mpimg.imsave(imagePath, images)


def main_function():  # Go through all dirs and call a function that mixes both photos
    for dirpath, dirnames, files in os.walk('image_dir\\'):
        print(dirpath)
        counter = 0
        color_image = None
        depth_image = None
        for file_name in files:
            if counter == 1:
                depth_image = mpimg.imread(dirpath + '\\' + file_name)
                final_name = file_name.replace('Depth', 'Combined')
                rgbd_function(color_image, depth_image,
                              dest_dir_path + dirpath.replace('image_dir\\', '') + '\\' + final_name)
                counter = 0
            else:
                color_image = mpimg.imread(dirpath + '\\' + file_name)
                counter = 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_function()
