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

public:
    PhotoTaker();
    void takePicture(int number);
};


#endif //PHOTO_TAKER_PHOTOTAKER_H
