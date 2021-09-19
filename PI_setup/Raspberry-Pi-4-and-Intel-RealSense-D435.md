## Pre-install Requirements
* Start with updating, upgrading, and installing dependencies and tools:
```
sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install automake libtool vim cmake libusb-1.0-0-dev libx11-dev xorg-dev libglu1-mesa-dev
```
* Expand the filesystem by selecting the `Advanced Options` menu entry, and select yes to rebooting:
```
sudo raspi-config
```
* Increase swap to 2GB by changing the file below to `CONF_SWAPSIZE=2048`:
```
sudo vi /etc/dphys-swapfile
```
* Apply the change: 
```
sudo /etc/init.d/dphys-swapfile restart swapon -s
```
* Create a new `udev` rule:
```
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 
```
* Apply the change (needs to be run by root):
```
sudo su
udevadm control --reload-rules && udevadm trigger
exit
```
* Modify the path by adding the following line to the `.bashrc` file:
```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```
* Apply the change:
```
source ~/.bashrc
```
## Installation
* Install `protobuf` — Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data:
```
cd ~
git clone --depth=1 -b v3.10.0 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure
make -j1
sudo make install
cd python
export LD_LIBRARY_PATH=../src/.libs
python3 setup.py build --cpp_implementation 
python3 setup.py test --cpp_implementation
sudo python3 setup.py install --cpp_implementation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
sudo ldconfig
protoc --version
```
* Install `libtbb-dev` parallelism library for C++:
```
cd ~
wget https://github.com/PINTO0309/TBBonARMv7/raw/master/libtbb-dev_2018U2_armhf.deb
sudo dpkg -i ~/libtbb-dev_2018U2_armhf.deb
sudo ldconfig
rm libtbb-dev_2018U2_armhf.deb
```
* Install RealSense SDK `librealsense`:
```
cd ~/librealsense
mkdir  build  && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release -DFORCE_LIBUVC=true
make -j1
sudo make install
```
* Install RealSense SDK `pyrealsense2` Python bindings for `librealsense`:
```
cd ~/librealsense/build
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)
make -j1
sudo make install
```
* Modify the path by adding the following line to the `.bashrc` file:
```
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
```
* Apply the change:
```
source ~/.bashrc
```
* Install `OpenGL`:
```
sudo apt-get install python-opengl
sudo -H pip3 install pyopengl
sudo -H pip3 install pyopengl_accelerate==3.1.3rc1
```
* Change pi settings (enable `OpenGL`):
```
sudo raspi-config
"7. Advanced Options" – "A8 GL Driver" – "G2 GL (Fake KMS)"
```
## Remote Operation
* Enable `VNC`:
```
"3. Boot Options" – "B1 Desktop/CLI" – "B4 Desktop Autologin"
"5. Interfacing Options" – "P3 VNC" – "Yes"
"7. Advanced Options" – "A5 Resolution" – "DMT Mode 85 1280X720 60Hz 16:9"
```
* Select "Finish" on the `raspi-config` menu, and select "Yes" to rebooting

* Install a VNC client on another computer, for example [VNC Viewer by RealVNC](https://www.realvnc.com/en/connect/download/viewer/)

* VNC into the Raspberry Pi and run the built-in viewer from a terminal window:
```
realsense-viewer
```
![Image of Intel's RealSense Viewer](https://github.com/acrobotic/Ai_Demos_RPi/blob/master/realsense/img/viewer.png)
## Debugging
* ERROR: [Could not initialize offscreen context!](https://github.com/acrobotic/Ai_Demos_RPi/issues/1#issue-628919760)
```
pi@raspberrypi:~ $ realsense-viewer
Could not initialize offscreen context!
```
There is a difference between running vncserver from the terminal and using raspi-config to install it. So, if you see the below output you're doing it wrong:
```
pi@raspberrypi:~ $ glxgears -info
GL_RENDERER = llvmpipe (LLVM 9.0.1, 128 bits)
GL_VERSION = 3.1 Mesa 19.3.2
GL_VENDOR = VMware, Inc.
```
Output known to be working:
```
GL_RENDERER = V3D 4.2
GL_VERSION = 2.1 Mesa 19.3.2
GL_VENDOR = Broadcom
```
The process to make the necessary changes is described in [raspberrypi/Raspberry-Pi-OS-64bit#21](https://github.com/raspberrypi/Raspberry-Pi-OS-64bit/issues/21) and [raspberrypi/Raspberry-Pi-OS-64bit#3](https://github.com/raspberrypi/Raspberry-Pi-OS-64bit/issues/3). It comes down to how we run `vncserver`, running it from a Putty terminal uses a **VMware** driver, whereas running it via `sudo raspi-config` > `interfacing options` > `vncserver` > `enable` and directly login from the VNC client app then uses the necessary **Broadcom** driver.

* ERROR: [GLFW Driver Error: GLX: GLX version 1.3 is required](https://github.com/acrobotic/Ai_Demos_RPi/issues/1#issue-628919760)
```
pi@raspberrypi:~ $ realsense-viewer
GLFW Driver Error: GLX: GLX version 1.3 is required
Could not initialize offscreen context!
```
Follow the steps described in [IntelRealSense/librealsense#7343](https://github.com/IntelRealSense/librealsense/issues/7343#issue-701858461).