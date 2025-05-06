# Volume-Estimation
This project integrates the [Heinsight vial detection model](https://gitlab.com/heingroup/heinsight4.0) with a live lab video feed, enabling real-time vial analysis for laboratory automation on edge computing devices like Jetson Nano. The model processes frames directly from the lab camera stream and is designed to be compatible with the laboratory automation system for further integration into robotic workflows.

## Deployment Instruction
### 1. Clone the Repository
```
git clone https://github.com/Joy12343/Volume-Estimation.git
cd Volume-Estimation
```

### 2. Set Up the Environment
Pull the latest Ultralytics Jetson-compatible Docker image and launch it with access to your project directory:
```
t=ultralytics/ultralytics:latest-jetson-jetpack4
sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia --volume ~/Volume-Estimation:/Volume-Estimation $t
```
Install dependencies inside the container:
```
pip install flask opencv-python
```
