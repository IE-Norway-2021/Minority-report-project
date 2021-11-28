docker run --gpus all -t -d --shm-size=1g --ulimit memlock=-1 --name gesture_ml_container ml/gesture
docker exec -it gesture_ml_container /bin/bash