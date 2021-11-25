sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install automake libtool nano cmake libusb-1.0-0-dev libx11-dev xorg-dev libglu1-mesa-dev libssl-dev libdrm-amdgpu1 libdrm-amdgpu1-dbgsym libdrm-dev libdrm-exynos1 libdrm-exynos1-dbgsym libdrm-freedreno1 libdrm-freedreno1-dbgsym libdrm-nouveau2 libdrm-nouveau2-dbgsym libdrm-omap1 libdrm-omap1-dbgsym libdrm-radeon1 libdrm-radeon1-dbgsym libdrm-tegra0 libdrm-tegra0-dbgsym libdrm2 libdrm2-dbgsym libglu1-mesa libglu1-mesa-dev glusterfs-common libglu1-mesa libglu1-mesa-dev libglui-dev libglui2c2 libglu1-mesa libglu1-mesa-dev mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev

sudo cp dphys-swapfile /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile restart swapon -s
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 

echo "enter the following command as su : udevadm control --reload-rules && udevadm trigger"
echo "Then type : \"exit\" and run ./setup_2.sh"
sudo su