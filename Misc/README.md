# Misc folder

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

## Dataset

The folder Dataset contains all code used for building the Dataset. It is split in two, one folder containing the code used for the picture dataset, the other for the video dataset.

### Photo : 

This folder contains the following directories/files : 

```bash
```

- The folder CPP_code contains the tests made with the realsense2 library for C++, and the code used to build the Picture dataset.
- The folder Dataset_handlers contains the script files used for the handling of the dataset (for counting the progress, separating the rgb from the depth,...)
- The three remaining python files were used in the process of finding the best format of image to use in the machine learning models.

For more information on each file, please read the documentation in the files themselves.

### Video : 


This folder contains the following directories/files : 

```bash
```



- The folder Driver contains test of ways to implement the driver for the main app.
