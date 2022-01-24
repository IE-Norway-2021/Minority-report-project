# Docker image directory

This folder contains the files needed to build a docker image with tensorflow gpu to train the models.

Docker image launching steps :
- have the video dataset unzipped in the ml folder, and have the ml.py in that folder as well
- launch the build_image.sh script to build image
- launch the start_container.sh script to start a container

Additionaly, there are two other scripts available : 
- join_created_container : allows to go into the container to laucnh the ml, check the results,...
- copy_output : allows to copy the output folder from inside the container