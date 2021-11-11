# Instalation guide for tensorflow gpu on super computer UIA

David González León

To install tensorflow gpu in the super computer, please follow the following instructions : 

- install virtual gpu driver : 
  ```bash
  # Uninstall cuda-drivers
  apt purge cuda-drivers
  apt autoremove
  # Install vgpu-driver
  curl -s -o /tmp/NVIDIA-Linux-x86_64.run http://128.39.54.48/downloads/NVIDIA-Linux-x86_64-470.63.01-grid.run
  /tmp/NVIDIA-Linux-x86_64.run --silent --no-drm -Z
  # Install cuda 11.4
  apt install cuda-minimal-build-11-4
  # Verify nvcc version
  /usr/local/cuda-11.4/bin/nvcc --version
  #verify nvidia driver is running
  nvidia-smi
  ```
- Download cudnn corresponding to CUDA 11.4, CUDNN v8.2.4
- follow installation guide for cudnn (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) step 2.3.1
- run `pip install tensorflow==2.4.0 tensorflow-gpu==2.4.0`
- run following command : `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/include:/usr/local/cuda/lib64`
  if using anaconda, run following command aswell : `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/anaconda3/envs/tf_gpu/lib/:` (use sudo find . -name libcublas.so.11 from root folder to find the exact path )
- Test tensorflow recognizes the gpu by running the following code in a python venv : 
```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```
