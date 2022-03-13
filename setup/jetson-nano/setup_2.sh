echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

#update cmake
cd ~
sudo apt install -y libssl-dev libcurl4-openssl-dev
wget http://cmake.org/files/v3.13/cmake-3.13.0.tar.gz
tar -xpvf cmake-3.13.0.tar.gz
cd cmake-3.13.0/
./bootstrap --system-curl
make -j6
echo 'export PATH=~/cmake-3.13.0/bin:$PATH' >> ~/.bashrc
source ~/.bashrc


cd ~/librealsense
mkdir -p build  && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=/usr/bin/python3.6 -DBUILD_WITH_CUDA:bool=true -DFORCE_RSUSB_BACKEND=ON -DCAMKE_BUILD_TYPE=release -DCMAKE_CUDA_COMPILER="/usr/local/cuda-10.2/bin/nvcc"
make -j4
sudo make install
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.6/pyrealsense2' >> ~/.bashrc
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc

cd ~/Minority-report-project/setup/jetson-nano
sudo addgroup uinput
sudo adduser user uinput
sudo cp 98-uinput.rules /etc/udev/rules.d/98-uinput.rules

# Install Tensorflow
# Remove existing first
sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow
# install the dependencies (if not already onboard)
sudo apt-get install -y gfortran
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install -y libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install -y liblapack-dev
pip3 install Cython==0.29.21
# install h5py with Cython version 0.29.21 (± 6 min @1950 MHz)
pip3 install h5py==2.10.0
pip3 install -U testresources numpy
# upgrade setuptools 39.0.1 -> 53.0.0
pip3 install --upgrade setuptools
pip3 install pybind11 protobuf google-pasta
pip3 install -U six mock wheel requests gast
pip3 install keras_applications --no-deps
pip3 install keras_preprocessing --no-deps
# install gdown to download from Google drive
pip3 install gdown
# download the wheel
gdown https://drive.google.com/uc?id=1DLk4Tjs8Mjg919NkDnYg02zEnbbCAzOz
# install TensorFlow (± 12 min @1500 MHz)
pip3 install tensorflow-2.4.1-cp36-cp36m-linux_aarch64.whl