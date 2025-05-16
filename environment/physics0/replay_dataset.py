from simulator.packingGame import PackingGame
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IN_FOLDER = 'dataset/RHSOrganizedV21'
OUT_FOLDER = 'temp'

env = PackingGame(visual=True, reward_function='compactness')

df_results_episode = pd.DataFrame(columns=['filename', 'obj_packed', 'compactness_episode', 'success_episode'])
df_per_step = pd.DataFrame(columns=['filename', 'compactness_step', 'stability', 'success', 'time_elapsed'])

# for every file in the input folder
for filename in os.listdir(IN_FOLDER):
    # load the file
    with open(os.path.join(IN_FOLDER, filename)) as f:
        data = json.load(f)
    new_name = filename.split('.')[0]
    # check does the new_name folder exist inside the OUT_FOLDER
    if not os.path.exists(os.path.join(OUT_FOLDER, new_name)):
        os.makedirs(os.path.join(OUT_FOLDER, new_name))

    i = 0

    # reset the environment
    env.reset()

    # unpacked list is the objectName column
    env.unpacked_list = [obj['objectName'] for obj in data]

    done = False

    obj_packed = 0
    print('press enter for each step')
    # while not done
    i = 0
    while not done:
        # take an observation
        env.render(heightname=os.path.join(OUT_FOLDER, new_name, new_name + '_height_' + f'{i}.png'), colorname=os.path.join(OUT_FOLDER, new_name, new_name + '_color_' + f'{i}.png'))
        if i == 0:
            heightmap = env.space.heightmapC
            data[i]['heightmapBefore'] = env.prepare_observation(heightmap, env.projections[env.objects.tolist().index(data[i]['objectName'])]).tolist()
        else:
            data[i]['heightmapBefore'] = obs.tolist()
        
        # take an action
        targetC = data[i]['translation']
        rotation = data[i]['rotationOriginalEuler']

        image = mpimg.imread(OUT_FOLDER + '/' +  new_name  + '/' + new_name + '_height_' + f'{i}.png')
        # plt.axis('off')  # for some reason colorbar is showing (hide it!)
        plt.show()

        # take an action
        # action = np.zeros(3)
        # action[:2] = data[i]['translation'][:2]
        # action[2] = data[i]['rotationApproximateEuler'][2]

        # take an action three rotations
        # pose = np.zeros(6)
        # pose[:3] = data[i]['translation']
        # quaternion = data[i]['rotationApproximateQuaternion']
        # pose[3:] = R.from_quat(quaternion).as_euler('xyz', degrees=True)

        obs, done, success, compactness, stability = env.replay_step_with_metrics(targetC=targetC, rotation=rotation)
        # obs, done, success, compactness, stability, _ = env.step_with_metrics(action, data[i]['objectName'], actions_normal=True)

        if success == 1 or success == 0:
                obj_packed += 1

        df_this_step = pd.DataFrame({'filename': filename, 'compactness_step': compactness, 'stability':stability, 'success': success, 'time_elapsed':0}, index=[0])
        df_per_step = pd.concat([df_per_step, df_this_step], ignore_index=True)

        i += 1

    print(f'Filename: {filename}, Objects packed: {obj_packed}, Compactness: {compactness}, Success: {success}')
    df_this_episode = pd.DataFrame({'filename': filename, 'obj_packed': obj_packed, 'compactness_episode': compactness, 'success_episode': success}, index=[0])
    df_results_episode = pd.concat([df_results_episode, df_this_episode], ignore_index=True)
    
    # write the data to a new file up to the point of failure
    with open(os.path.join(OUT_FOLDER, new_name, new_name + 'data.json'), 'w') as f:
        json.dump(data[:i], f)

df_results_episode.to_csv(OUT_FOLDER + '/' + 'human_test_results_episode_check.csv', index=False)
df_per_step.to_csv(OUT_FOLDER + '/' + 'human_test_results_step_check.csv', index=False)

env.close()
