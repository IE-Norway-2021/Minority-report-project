# 1. ML folder

- [1. ML folder](#1-ml-folder)
  - [1.1. Docker image tf gpu :](#11-docker-image-tf-gpu-)
  - [1.2. ML gesture results :](#12-ml-gesture-results-)
  - [1.3. ML video results :](#13-ml-video-results-)
    - [1.3.1. Kfold :](#131-kfold-)
    - [1.3.2. Training](#132-training)
  - [1.4. test real time :](#14-test-real-time-)
  - [1.5. Root folder :](#15-root-folder-)
    - [1.5.1. ML.py :](#151-mlpy-)
    - [1.5.2. tf_lite_converter :](#152-tf_lite_converter-)

This folder contains all files related to the machine learning part of the application. This includes all the scripts used to create and train the models, and the results of the training of each main model

The directory is structured in the following way

```bash
.
├── Docker_image_tf_gpu
├── ML_gesture_results
├── ML_video_results
│   ├── Kfold
│   └── Training
├── test_real_time
├── ml.py
├── tf_lite_converter.ipynb
└── tf_lite_converter.py
```

## 1.1. Docker image tf GPU :

This folder contains the necessary files to create a docker image to run the training. For more information please read the [readme file](Docker_image_tf_gpu/README.md) located in the folder.

## 1.2. ML gesture results :  

This folder contains the results of training of the hand image dataset.

## 1.3. ML video results : 

This folder contains the results of the training of all video models. It is divided in 2 folders : 

### 1.3.1. Kfold : 

This folder contains the results of the kfold training of the 3 main models presented in the article, and the graphs used in the article relating to kfold training. 

### 1.3.2. Training

This folder contains all the results from training the 3 main models presented in the article. For each model there is the weight resulting from the training, and the history of the training session.

## 1.4. Test real time : 

This folder contains the code used to test the real time prediction. The main two files in this folder are [main_video.py](test_real_time/main_video.py), that contains the code used to a certain the test accuracy of the models, and [testing_real_time.py](test_real_time/testing_real_time.py), that contains the code used to create the final application, but without the inclusion of the driver.

## 1.5. Root folder : 

The root folder ML contains 3 very important files : 

### 1.5.1. [ML.py](ml.py) : 

This file contains all the functions that allow the training of the models, as well as the generation of various graphs based on the training history of the models.

### 1.5.2. tf_lite_converter : 

These two files contain the code used to create tf_lite weights for our models. In our case, the Raspberry pi 4 did not increase in effectiveness when using this technology, but it could be used on other platforms to increase the inference speed.