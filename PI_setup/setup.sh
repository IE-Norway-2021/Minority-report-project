sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install automake libtool nano cmake libusb-1.0-0-dev libx11-dev xorg-dev libglu1-mesa-dev libssl-dev 

sudo cp dphys-swapfile /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile restart swapon -s
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 

echo "enter the following command as su : udevadm control --reload-rules && udevadm trigger"
echo "Then type : \"exit\" and run ./setup_2.sh"
sudo su