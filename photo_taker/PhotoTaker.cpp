//
// Created by jaden on 20/09/2021.
//

#include "PhotoTaker.h"
#include <librealsense2/rs_advanced_mode.hpp>


PhotoTaker::PhotoTaker() {

}

void PhotoTaker::takePicture(int number){
    auto colorData = PXCImage::ImageData();

    if (image->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &colorData) >= PXC_STATUS_NO_ERROR) {
        auto colorInfo = image->QueryInfo();
        auto colorPitch = colorData.pitches[0] / sizeof(pxcBYTE);
        Gdiplus::Bitmap tBitMap(colorInfo.width, colorInfo.height, colorPitch, PixelFormat24bppRGB, baseColorAddress);
    }
}