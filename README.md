# HERB - Hierarchical Robot Learning

A Python implementation for hierarchical robot learning using SAC (Soft Actor-Critic).

## 📝 Description
HERB is a framework for training robotic agents using hierarchical reinforcement learning approaches.

## 🔽 Download

You can download the latest model from the [Release page](https://github.com/NunoDuarte/herb/releases/latest)  
Or directly: [Download model](https://github.com/NunoDuarte/herb/releases/download/v1.0/sac_model_c09s01.pkl)

## 🛠️ Installation

```bash
git clone https://github.com/NunoDuarte/herb.git
cd herb
pip install -r requirements.txt
```

## 🤖 ROS Integration

### Environment Setup
```bash
# Set ROS network configuration
export ROS_IP=<local_pc_ip>        # Your PC's IP address
export ROS_MASTER_URI=<baxter_ip>  # Baxter robot's IP address
```

### ROS Part
Don't forget to set ROS_IP (local PC) and ROS_MASTER_URI (Baxter PC) for all terminals.

#### First Terminal
Open RealSense camera:
```bash
source /opt/ros/noetic.sh
roslaunch realsense2_camera rs_camera.launch
```

#### Second Terminal
Process raw depth for cropped heightmap of box (top-view):
```bash
/usr/bin/python3 realsense_depth_process.py
```

#### Third Terminal
Run baxter_demo which sends heightmap to RL to predict place location of object on the box:
```bash
source /opt/ros/noetic.sh
/home/user/env/packbot/bin/python3 environment/physics0/baxter_demos.py
```



