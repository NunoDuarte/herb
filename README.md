# HERB - Hierarchical Robot Learning
HERB is a framework for training robotic agents using hierarchical reinforcement learning approaches. This work is based on our paper 
[arXiv](https://arxiv.org/pdf/2504.16595)

<img src="media/pybullet_gif.gif" width="233" height="300" /> <img src="media/iros25_herb_baxter.gif" width="535" height="300" />

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
Add python path
```bash
export PYTHONPATH=/PATH/TO/herb
```
### run human packing sequences in Pybullet
```bash
python environment/physics0/replay_dataset.py
```
### run trained RL policy for test set sequences
```bash
python environment/physics0/generate_testset_packs.py
```
you can specific which sequence by looking into file ```test_set_list.json```
### run trained RL policy for any sequence of objects
```bash
python environment/physics0/test_model.py
```
you can specify the objects by changing the list in ```unpacked_list```. 
## 🔽 Download pretrained model

You can download the latest model from the [Release page](https://github.com/NunoDuarte/herb/releases/latest)  
Or directly: [Download model](https://github.com/NunoDuarte/herb/releases/download/v1.0/sac_model_c04s06.pkl)

## Train RL 
TODO
Some arguments that can be changed for a different packing strategy/configuration: 
``` python 
bin_size=[0.345987, 0.227554, 0.1637639] # size of the box
object_info='dataset/datas/object_info.npz' , # information file about object projections, volumes, to make observations and calculate metrics
visual=False, # rendering
ordered_objs=False, # to use Beam-3 to order the list of objects or no
reward_function='simple', # or 'compactness' or 'comapctness_stability'
alpha=0.9, # trade off between compactness and stability if that reward is used
unpacked_list_min=0.7, unpacked_list_max=0.9 # the parameters by which (BioRob2024)[https://arxiv.org/abs/2210.01645] generated objects for the pack, the sum of object volumes to be packed is between 0.7 to 0.9 (fixed value)
```

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

### ROS master
TODO: add instructions to run baxter and the demo to move robot

### ROS + RL Demo
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
