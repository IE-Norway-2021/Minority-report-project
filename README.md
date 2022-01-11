# 1. README

This project is inspired by the movie Minority Report. Using a depth camera and light weight CNN, it recognizes some specific gesture and converts it to Linux input events. For example, this project can be used to control application like Google Earth with hand gesture. 

- [1. README](#1-readme)
  - [1.1. Installation](#11-installation)
  - [1.2. Repository structure](#12-repository-structure)
  - [1.3. Article about the project](#13-article-about-the-project)
  - [1.4. Using the application](#14-using-the-application)

## 1.1. Installation 

The hardware needed is :

- Raspberry pi 4 with Raspbian OS
- Intel Realsense depth camera D435

The installation procedure for the Raspberry Pi is described in the [readme](PI_setup/README.md) of the PI_setup folder.  

## 1.2. Repository structure

There are 4 folders in the project's root. The folders [Misc](Misc/README.md), [PI_setup](PI_setup/README.md) and [ML](ML/README.md) already contain readme describing the files in it. 

The src folder contains the main code of the project. 

## 1.3. Article about the project

This project was the main focus of an article that will soon be published in the IEEE Sensors journal. 

## 1.4. Using the application

To launch the application, first go to the [uinput folder](src/uinput) and run the following command `make module`. Then go back to the root of the src folder and run the real_time_app.py script (`python real_time_app.py`).

To make sure the application will work you need to place the weights used in the app at the root of the src folder, and have a D435 camera connected to the raspberry pi.