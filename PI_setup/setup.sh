echo 'Have you expanded the filesystem in the advanced options (y/n)?'
read var
if [[ "$var" = "n" ]]; then
echo 'Please expand the filesystem first (sudo raspi config then advanced options)'
else
sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install -y automake libtool nano cmake libusb-1.0-0-dev libx11-dev xorg-dev libglu1-mesa-dev libssl-dev
sudo nano /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile restart swapon -s
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 
sudo su
udevadm control --reload-rules && udevadm trigger
exit
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source ~/.bashrc
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
cd ~
wget https://github.com/PINTO0309/TBBonARMv7/raw/master/libtbb-dev_2018U2_armhf.deb
sudo dpkg -i ~/libtbb-dev_2018U2_armhf.deb
sudo ldconfig
rm libtbb-dev_2018U2_armhf.deb
cd ~/librealsense
mkdir  build  && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release -DFORCE_LIBUVC=true
make -j1
sudo make install
cd ~/librealsense/build
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)
make -j1
sudo make install
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
source ~/.bashrc
sudo apt-get install -y python-opengl
sudo -H pip3 install pyopengl
sudo -H pip3 install pyopengl_accelerate==3.1.3rc1
echo 'Now please enable the OpenGL ine the advanced options (sudo raspi-config)'
fi