# HERB - Hierarchical Robot Learning
HERB is a framework for training robotic agents using hierarchical reinforcement learning approaches. This work is based on our paper 
[arXiv](https://arxiv.org/pdf/2504.16595)

## 🛠️ Installation

```bash
git clone https://github.com/NunoDuarte/herb.git
cd herb
pip install -r requirements.txt
```
### 🔧 Tested Environment
- **OS**: Ubuntu 20.04 LTS
- **Python**: 3.8
- **CUDA**: 12.0

## Pybullet Simulation
### run trained RL policy for packing
```bash
export PYTHONPATH=/PATH/TO/herb
python environment/physics0/test_model.py
```

## 🔽 Download pretrained model

You can download the latest model from the [Release page](https://github.com/NunoDuarte/herb/releases/latest)  
Or directly: [Download model](https://github.com/NunoDuarte/herb/releases/download/v1.0/sac_model_c09s01.pkl)

## 🤖 ROS Integration
We also run the RL policy on the Baxter robot 
### Prerequisites
- ROS Noetic (Ubuntu 20.04)
- Intel RealSense SDK 2.0
- Python 3.8+
- Baxter SDK
- Additional ROS packages:
  ```bash
  sudo apt install ros-noetic-realsense2-camera
  sudo apt install ros-noetic-realsense2-description

### Environment Setup
```bash
# Set ROS network configuration
export ROS_IP=<local_pc_ip>        # Your PC's IP address
export ROS_MASTER_URI=<baxter_ip>  # Baxter robot's IP address
```

### Demo
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

## Troubleshooting Guide
#### Camera Issues
```bash
# Check if RealSense camera is detected
rs-enumerate-devices

# Test camera stream
realsense-viewer

# Verify camera topics
rostopic list | grep camera
```
#### Baxter Issues
```bash
# Test Baxter connectivity
ping <baxter_ip>

# Check Baxter status
rosrun baxter_tools enable_robot.py -s

# Verify joint states
rostopic echo /robot/joint_states

# Check ROS environment variables
echo $ROS_MASTER_URI
echo $ROS_IP
```

## 📄 Citation

If you find this code useful, please cite:
```bibtex
@article{perovic2025herb,
  title={HERB: Human-augmented Efficient Reinforcement learning for Bin-packing},
  author={Perovic, Gojko and Duarte, Nuno Ferreira and Dehban, Atabak and Teixeira, Gon{\c{c}}alo and Falotico, Egidio and Santos-Victor, Jos{\'e}},
  journal={arXiv preprint arXiv:2504.16595},
  year={2025}
}
```
