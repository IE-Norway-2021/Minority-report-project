FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 nano  -y
RUN pip install opencv-python pillow matplotlib scikit-learn

# The folder ml needs to contain the video_dataset folder and the ml.py file
COPY ml /ml 
COPY src /src

CMD ["python", "/src/main.py"]

