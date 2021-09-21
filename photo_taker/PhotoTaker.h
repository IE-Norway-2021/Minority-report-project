//
// Created by jaden on 20/09/2021.
//

#ifndef PHOTO_TAKER_PHOTOTAKER_H
#define PHOTO_TAKER_PHOTOTAKER_H

#include <librealsense2/rs.hpp>
#include <fstream>              // File IO
#include <iostream>             // Terminal IO
#include <sstream>              // Stringstreams

class PhotoTaker {
private:
    unsigned int fileNo = 0;
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;


public:
    PhotoTaker();
    void takePicture(int number);

};


#endif //PHOTO_TAKER_PHOTOTAKER_H
