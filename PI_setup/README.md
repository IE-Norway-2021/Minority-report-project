# Installation

Before starting you need to have the following to be able to complete the installation : 

- A raspberry pi with Raspbian OS (desktop version) (a clean instalation)
- An internet connection

# Pre installation requirements : 

- If you haven't, download the repo by cloning it on you raspberry pi, or by taking the latest available release 

- Expand the file system by selecting the option in the `Advanced Options` tab and select yes to reboot

```bash
sudo raspi-config
```

- Then execute the following script (located in the PI_setup folder): 

```bash
./setup.sh
```

- Once the script finishes running you should be in su mode. You need to restart the dev rules : 

```bash
udevadm control --reload-rules && udevadm trigger
exit
```

## Installation

- Run the following script located in the PI_setup folder (will take a long time to finish) : 

```bash
./setup_2.sh
```

- Finally restart the device

```bash
sudo reboot 0
```