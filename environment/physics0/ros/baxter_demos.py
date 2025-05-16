import numpy as np
import cv2
import matplotlib.pyplot as plt
from environment.physics0.simulator.space import draw_heatmap, draw_heatmap_box

object_info_path = 'dataset/datas/object_info.npz'
object_info = np.load(object_info_path)
projections = object_info['projections']
ALL_OBJECTS = [
    "002 MasterChef Can",
    "003 Cracker Box",
    "004 Sugar Box",
    "005 Tomato Coup Can",
    "006 Mustard Bottle",
    "007 Tuna Fish Can",
    "008 Pudding Box",
    "010 Potted Meat Can",
    "011 Banana",
    "012 Strawberry",
    "013 Apple",
    "014 Lemon",
    "015 Peach",
    "016 Pear",
    "017 Orange",
    "018 Plum",
    "021 Bleach Cleanser",
    "025 Mug",
    "057 Racquetball",
    "058 Golf Ball",
    "100 Half Egg Carton",
    "101 Bread",
    "102 toothbrush",
    "103 toothpaste",
]
PROJECTIONS_DICT = dict(zip(ALL_OBJECTS, projections))

# unnormalize from [-1 to 1]
def unnormalize(x, min, max):
    return 0.5 * (x + 1) * (max - min) + min

# draw the heatmap
def draw_heatmap_norm(heightMap, vmin = 0, vmax = 255, save = False, savename='test.png', show = True):
    # close previous figure
    plt.close()
    # draw new figure
    plt.figure()
    plt.imshow(heightMap,  cmap=plt.cm.hot, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if save:
        plt.savefig(savename)
    if show:
        plt.show()

def prepare_obs_camera(image, obj_name, camera_offset, eps=0.05):
    """
    takes an image and returns a 1x224x224 compatible image
    """

    # (1) remove the camera offset so the table is at 0
    image = image - camera_offset
    image = image[5:, 10:]
    # (1a) possibly filter if it does not look ok with some threshold
    image[image < eps] = 0

    # (2) make sure the image is 115x76, if not resize it
    if image.shape != (115, 76):
        image = cv2.resize(image, (76, 115))
        
    image = np.flip(image)
    
    # (3) if necessary mask some rows or columns with 0 to remove the sides of the box

    # (4) standard processing from the environment
    # scale the heightmap from [0, 0.3] to [0, 255]
    heightmap = (image / 0.3) * 255
    projection = PROJECTIONS_DICT[obj_name]
    
    # pad the heightmap from 115x76 to 224x224
    padded_heightmap = np.zeros((224, 224))
    padded_heightmap[54:54+heightmap.shape[0], 26:26+heightmap.shape[1]] = heightmap

    # add the projection to the padded heightmap
    padded_heightmap[76:76+projection.shape[0], 100+26:(100+26+projection.shape[1])] = projection

    # convert the padded heightmap to uint8
    padded_heightmap = padded_heightmap.astype(np.uint8)

    # convert the padded heightmap to [1 x 224 x 224]
    padded_heightmap = np.expand_dims(padded_heightmap, axis=0)

    return padded_heightmap

if __name__ == '__main__':
    from stable_baselines3.sac.policies import CnnPolicy
    import pandas as pd
    from packingGame import PackingGame
    import subprocess
    import time
    import json

    # import test set list json
    with open('test_set_list.json') as f:
        test_set_list = json.load(f)

    subprocess.run('source /opt/ros/noetic/setup.bash', shell=True)
    subprocess.run('/usr/bin/python3 ~/gojko-irbpp/environment/physics0/ros/realsense_depth_process.py', shell=True)
        
    time.sleep(3)

    # load the model
    model_pth = 'policies/0221_for_human_comparison.pkl'
    policy = CnnPolicy.load(model_pth)
    policy.eval()
    print('Model loaded')
    
    experiment = 'p24_s1'

    # load the unpacked_list or create it
    unpacked_list = test_set_list[experiment]


    # laod the image, or take it directly from the camera
    image = 0
    image = np.load('environment/physics0/images/heightmap.npy') / 1000.0
    print(np.min(image))
    print(np.max(image))

    
    # prepare the dataframe
    df = pd.DataFrame(columns=['objectName', 'x', 'y', 'theta'])

    terminated = False

    env = PackingGame(visual=True, ordered_objs=True)

    obs_simulated, _ = env.reset(unpacked_list=unpacked_list)
    unpacked_list = env.unpacked_list
    print('Unpacked list', unpacked_list)

    # prepare the first observation
    # get the object name
    # Retirar - sequencia: objname = unpacked_list[0] !!!!!
    objname = unpacked_list[6]
    obs = prepare_obs_camera(image, objname, camera_offset=0.0)
    print('Object: ', objname)
    input('Press button to start the loop')
    while not terminated:

        
        # get the action from the policy
        action, _ = policy.predict(obs, deterministic=True)
        action_simulated = action.copy()

        obs_simulated, _, _, _, _ = env.step(action_simulated)

        # unnormalize the action
        x = unnormalize(action[0], 0.0, 0.345987)
        y = unnormalize(action[1], 0.0, 0.227554)
        theta = unnormalize(action[2], 0.0, 180.0)

        # append to the dataframe
        df_this_step = pd.DataFrame({'objectName': [objname], 'x': [x], 'y': [y], 'theta': [theta]})
        df = pd.concat([df, df_this_step], ignore_index=True)

        # print the action
        print('X:', x, 'Y:', y, 'Theta:', theta)

        with open("environment/physics0/images/pose.txt", "w") as f:
            f.write(f"{x},{y},{theta}")

        import os

        # Set the ROS environment variables manually
        os.environ['ROS_MASTER_URI'] = 'http://baxter-vislab.local:11311'
        os.environ['ROS_IP'] = '10.16.145.134'
        subprocess.run('source /opt/ros/noetic/setup.bash', shell=True)
        subprocess.run('/usr/bin/python3 ~/gojko-irbpp/environment/physics0/ros/baxter_ros_comm.py', shell=True)

        # check is the action successful
        button = input('Continue (y/n)?')
        if button == 'n':
            terminated = True
            df.to_csv('TEMP.csv')
            break
        
        subprocess.run('source /opt/ros/noetic/setup.bash', shell=True)
        subprocess.run('/usr/bin/python3 ~/gojko-irbpp/environment/physics0/realsense_depth_process.py', shell=True)
        
        time.sleep(3)

        # prepare the next observation
        objname = unpacked_list[0]

        # check is the packing finished
        if len(unpacked_list) == 0:
            print('Packing finished!')
            terminated = True
            df.to_csv('TEMP.csv')
            break

        

        # load the image, or take it directly from the camera
        image = np.load('environment/physics0/images/heightmap.npy') / 1000.0

        obs = prepare_obs_camera(image, objname, camera_offset=0.0)
        print('Object: ', objname)
        draw_heatmap_norm(obs[0])




