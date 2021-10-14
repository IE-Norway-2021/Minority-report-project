import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

root_name = 'Dataset'
format = np.array(['rgb', 'depth'])
actions = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'start', 'stop'])
no_sequences = 20
sequence_length = 40


def video_taker():
    # Setup pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    print('starting the program...')
    while True:

        print("Choose num gesture or press q to quit : ")
        for index in range(actions.size):
            print('{} : {}'.format(actions[index], index))
        action = input()
        if action == 'q':
            print('Quitting now...')
            return
        if (not action.isdigit()) or int(action) >= actions.size or int(action) < 0:
            print('Wrong input please try again')
            continue
        action = int(action)
        # gesture chosen, doing the video taking

        for sequence in range(no_sequences):
            now = datetime.datetime.now()
            sequence_folder_name = now.strftime("%Y-%m-%d-%H%M%S") + '_{}'.format(sequence)
            os.makedirs(os.path.join(root_name, 'rgb', actions[action], sequence_folder_name))
            os.makedirs(os.path.join(root_name, 'depth', actions[action], sequence_folder_name))
            input(f"Press enter to start collection {sequence + 1}")
            for frame_num in range(sequence_length):
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    return
                colorizer = rs.colorizer()
                colorizer.set_option(rs.option.color_scheme, 0)
                # Convert images to numpy arrays
                depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # save image
                cv2.imwrite(f'{root_name}/rgb/{actions[action]}/{sequence_folder_name}/{frame_num}.png', color_image)
                cv2.imwrite(f'{root_name}/depth/{actions[action]}/{sequence_folder_name}/{frame_num}.png', depth_image)
            print('sequence finished')
    pipeline.stop()


# sequence folder name 21-10-12-hh-mm_num
# NEW LOOP
# Loop through actions
# for action in actions:
#     # Loop through sequences aka videos
#     for sequence in range(no_sequences):
#         # Loop through video length aka sequence length
#         for frame_num in range(sequence_length):


if __name__ == '__main__':
    video_taker()
