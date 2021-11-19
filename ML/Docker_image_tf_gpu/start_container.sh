docker run --gpus all -t -d --name gesture_ml_container ml/gesture 
docker exec -it gesture_ml_container /bin/bash