""" folder_creator
This script creates the dataset folder. It contains 2 folders for the rgb and the depth sequences (format array). 
In each of those file, there are as many files as there are gestures in the actions array.
The root name needs to be changed as needed. If there is already a folder with the same name it runs into an error without
impacting the folder that already exists. 
"""

import os
import numpy as np

root_name = 'tmp'
format = np.array(['rgb', 'depth'])
actions = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out'])


def folder_creator():
    for type in format:
        for action in actions:
            os.makedirs(os.path.join(root_name, type, action))


if __name__ == '__main__':
    folder_creator()
