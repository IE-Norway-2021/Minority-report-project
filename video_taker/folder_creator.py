import os
import numpy as np

root_name = 'video_dataset'
format = np.array(['rgb', 'depth'])
actions = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'start', 'stop'])

def folder_creator():
    for type in format:
        os.makedirs(os.path.join(root_name, type))
        for action in actions:
            os.makedirs(os.path.join(root_name, type, action))


if __name__ == '__main__':
    folder_creator()
