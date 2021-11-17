FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
   apt-get install --no-install-recommends ffmpeg libsm6 libxext6 nano -y && \
   apt clean && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip && \
   python -m pip install --no-cache-dir opencv-python==4.5.4.58 pillow==8.4.0 matplotlib==3.5.0 scikit-learn==1.0.1

# The folder ml needs to contain the video_dataset folder and the ml.py file
COPY /ml /ml 
