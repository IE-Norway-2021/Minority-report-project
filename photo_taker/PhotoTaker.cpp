//
// Created by jaden on 20/09/2021.
//

#include "PhotoTaker.h"
#include <librealsense2/rs_advanced_mode.hpp>
#include <fstream>              // File IO
#include <iostream>             // Terminal IO
#include <sstream>              // Stringstreams
#include <sys/stat.h>
#include <string>
// 3rd party header for writing png files
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include <dirent.h>
#include <time.h>

bool DirectoryExists(const char *pzPath) {
    if (pzPath == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir(pzPath);

    if (pDir != NULL) {
        bExists = true;
        (void) closedir(pDir);
    }

    return bExists;
}

PhotoTaker::PhotoTaker() {
    std::string imageFolder = "Images";
    if (!DirectoryExists(imageFolder.c_str())) {
        mkdir(imageFolder.c_str(), 0777);
    }
    for (int i = 0; i < 10; ++i) {
        std::string finalDir = (imageFolder + "/" + std::to_string(i));
        if (!DirectoryExists(finalDir.c_str())) {
            mkdir(finalDir.c_str(), 0777);
        }
    }
}

void PhotoTaker::takePicture(int number) try {

    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    // Capture 30 frames to give autoexposure, etc. a chance to settle
    //for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();

    // Wait for the next set of frames from the camera. Now that autoexposure, etc.
    // has settled, we will write these to disk

    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 80, "%F-%T", timeinfo);

    for (auto &&frame: pipe.wait_for_frames()) {

        // We can only save video frames as pngs, so we skip the rest
        if (auto vf = frame.as<rs2::video_frame>()) {
            auto stream = frame.get_profile().stream_type();
            // Use the colorizer to get an rgb image for the depth stream
            if (vf.is<rs2::depth_frame>()) vf = color_map.process(frame);

            // Write images to disk
            std::stringstream png_file;
            png_file << "Images/" << number << "/" << buffer << "_" << vf.get_profile().stream_name() << ".png";
            stbi_write_png(png_file.str().c_str(), vf.get_width(), vf.get_height(),
                           vf.get_bytes_per_pixel(), vf.get_data(), vf.get_stride_in_bytes());
            std::cout << "Saved " << png_file.str() << std::endl;
        }
    }
}
catch (const rs2::error &e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    "
              << e.what() << std::endl;
}
catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
}

void metadata_to_csv(const rs2::frame &frm, const std::string &filename) {
    std::ofstream csv;

    csv.open(filename);

    //    std::cout << "Writing metadata to " << filename << endl;
    csv << "Stream," << rs2_stream_to_string(frm.get_profile().stream_type()) << "\nMetadata Attribute,Value\n";

    // Record all the available metadata attributes
    for (size_t i = 0; i < RS2_FRAME_METADATA_COUNT; i++) {
        if (frm.supports_frame_metadata((rs2_frame_metadata_value) i)) {
            csv << rs2_frame_metadata_to_string((rs2_frame_metadata_value) i) << ","
                << frm.get_frame_metadata((rs2_frame_metadata_value) i) << "\n";
        }
    }

    csv.close();
}