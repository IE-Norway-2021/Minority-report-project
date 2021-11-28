sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install -y automake libtool nano cmake libusb-1.0-0-dev libx11-dev xorg-dev libglu1-mesa-dev libssl-dev gcc-10-arm-linux-gnueabi

sudo apt-get install -y libdrm-amdgpu1 libdrm-dev libdrm-exynos1 libdrm-freedreno1 libdrm-nouveau2 libdrm-omap1 libdrm-radeon1 libdrm-tegra0 libdrm2
sudo apt-get install -y libglu1-mesa libglu1-mesa-dev glusterfs-common libglu1-mesa libglu1-mesa-dev
sudo apt-get install -y libglu1-mesa libglu1-mesa-dev mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev

sudo cp dphys-swapfile /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile restart swapon -s
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 

echo "enter the following command as su : udevadm control --reload-rules && udevadm trigger"
echo "Then type : \"exit\" and run ./setup_2.sh"
sudo su