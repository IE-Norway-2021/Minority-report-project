# 1. Misc folder

- [1. Misc folder](#1-misc-folder)
  - [1.1. Dataset :](#11-dataset-)
    - [1.1.1. Photo :](#111-photo-)
    - [1.1.2. Video :](#112-video-)
  - [1.2. Driver](#12-driver)

This folder contains miscelanious files used while developping the main app. The structure and content of this folder and sub-folders is explained here.

The directory is structured in the following way

```bash
.
├── Dataset
│   ├── photo
│   └── video
└── Driver
```

Bellow is an explanation of the use of each file/directory.

## 1.1. Dataset :

The folder Dataset contains all code used for building the Dataset. It is split in two, one folder containing the code used for the picture dataset, the other for the video dataset.

### 1.1.1. Photo : 

This folder contains the following directories/files : 

```bash
.
├── CPP_code
│   ├── code_test
│   └── photo_taker
├── Dataset_handlers
│   ├── mkdir.sh
│   ├── mover.sh
│   └── progression_counter.sh
├── landmark_drawer.py
├── photo_mixer.py
└── redim_image.py
```

- The folder CPP_code contains the tests made with the realsense2 library for C++, and the code used to build the Picture dataset.
- The folder Dataset_handlers contains the script files used for the handling of the dataset (for counting the progress, separating the rgb from the depth,...)
- The three remaining python files were used in the process of finding the best format of image to use in the machine learning models.

For more information on each file, please read the documentation in the files themselves.

### 1.1.2. Video : 

This folder contains the following directories/files : 

```bash
.
├── folder_creator.py
├── image_resizer.py
├── landmark_drawer_video.py
├── progression_counter.sh
└── video_taker.py
```

The python files contain the code used for the creation of the dataset. The file [video_taker](Dataset/video/video_taker.py) is the most important, since it contains the code used for taking the sequences that form the dataset. 

For more information on each file, please read the documentation in the files themselves.

## 1.2. Driver

This folder contains the following directories/files : 

```bash
.
├── procfs_test
└── uinput_libevdev
```

The folder Driver contains tests of ways to implement the driver for the main app. 

- the folder procfs_test contains a small test of a driver using the procfs module
- The folder uinput_libevdev contains a test of a python module using the libevdev library to access the uinput api.

