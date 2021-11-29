echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz
tar -zxvf Python-3.7.12.tgz
mv Python-3.7.12 Python
rm Python-3.7.12.tgz
cd Python
./configure --enable-optimizations
sudo make altinstall
cd /usr/bin
sudo rm python
sudo ln -s /usr/local/bin/python3.7m python
python --version
sudo rm python3
sudo ln -s /usr/local/bin/python3.7m python3
python --version

cd ~
git clone --depth=1 -b v3.10.0 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure
make -j1
sudo make install
cd python
export LD_LIBRARY_PATH=../src/.libs
python setup.py build --cpp_implementation 
python setup.py test --cpp_implementation
sudo python setup.py install --cpp_implementation
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
cmake .. -DBUILD_EXAMPLES=false -DCMAKE_BUILD_TYPE=Release -DFORCE_LIBUVC=true
make -j1
sudo make install
cd ~/librealsense/build
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)
make -j1
sudo make install
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.7/dist-packages/pyrealsense2
source ~/.bashrc

cd ~/Minority-report-project/PI_setup/
sudo addgroup uinput
sudo adduser pi uinput
sudo cp 98-uinput.rules /etc/udev/rules.d/98-uinput.rules

sudo apt-get install pip raspberrypi-kernel-headers swig4.0
python -m pip install opencv-python matplotlib numpy sklearn pillow

sudo apt-get install -y python3-opengl
sudo python -m pip install pyopengl
sudo python -m pip install pyopengl_accelerate

sudo python -m pip install numpy
sudo apt-get install -y gfortran libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev
sudo python -m pip install --upgrade setuptools
sudo python -m pip install pybind11
sudo python -m pip install Cython
sudo python -m pip install h5py
python -m pip install gdown
gdown https://drive.google.com/uc?id=158xXoPWOyfNswDTaapyqpREq_CBk1O_G
sudo python -m pip install tensorflow-2.5.0-cp37-cp37m-linux_aarch64.whl
rm tensorflow-2.5.0-cp37-cp37m-linux_aarch64.whl