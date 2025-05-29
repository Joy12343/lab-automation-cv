# deck_detection
This module detects lab decks using a YOLOv8n-based object detection model. It was developed as part of a computer vision system for lab automation at Coley Research Group.

This folder contains all files related to the training and deployment of the deck detection model.

## Deployment Instruction
### 1. Pull the Jetson-Compatible Ultralytics
This includes the necessary YOLOv8 dependencies optimized for JetPack 4 (used on Jetson Nano and similar devices).
'''
t=ultralytics/ultralytics:latest-jetson-jetpack4
'''

### 2. Enable X11 Display Access
To allow GUI-based inference (e.g., showing images with bounding boxes):
'''
xhost +local:root
'''

### 3. Run the Container with Deck Detection Code Mounted
'''
sudo docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --env DISPLAY=$DISPLAY \
  --volume ~/deck_detection:/deck_detection \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  $t
'''

### 4. Run detect.py
'''
python3 detect.py
'''
