from environment.physics0.simulator.packingGame import PackingGame, draw_heatmap
import os
import json
import pandas as pd
import time
import numpy as np
from stable_baselines3.sac.policies import CnnPolicy


# create a tmp_path for the experiment
tmp_path = "PATH/TO/tmp/FOLDER/"
if tmp_path == "PATH/TO/tmp/FOLDER/":
    raise ValueError("Please change the tmp_path to your desired path. Edit the 'tmp_path' variable above.")

# check does the tmp_path exist
if not os.path.exists(tmp_path):
    # if not, create the directory
    os.makedirs(tmp_path)

model_path = 'PATH/TO/MODEL/.pkl'  # Change this to the path of your trained model
if not os.path.exists(model_path):
    raise ValueError(f"The model path '{model_path}' does not exist. Please check the path.")

TEST_FILE = 'dataset/test_set_list.json'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# open json file
with open(TEST_FILE) as f:
    test_set_list = json.load(f)
    
env = PackingGame(visual=True, ordered_objs=True, reward_function='compactness_stability', alpha=0.6, unpacked_list_min=0.7, unpacked_list_max=0.9)

policy = CnnPolicy.load(model_path)
policy.to('cuda')
# set policy to eval
policy.eval()

df_results_episode = pd.DataFrame(columns=['filename', 'obj_packed', 'compactness_episode', 'success_episode'])
df_per_step = pd.DataFrame(columns=['filename', 'compactness_step', 'stability', 'success', 'time_elapsed'])

time_elapsed_list = []

id = 4  # Change this to the number of sequences you want to test; if all sequences, set id to len(test_set_list)
pick_n_of_sequences = dict(list(test_set_list.items())[:id])
for filename in pick_n_of_sequences:
    print(filename)
    unpacked_list = test_set_list[filename]
    obs, _ = env.reset(unpacked_list=unpacked_list)
    done = False
    obj_packed = 0

    while not done:
        heightmap = obs[0]
        # unnormalize heightmap
        heightmap = (heightmap / 255.0) * 0.3 
        draw_heatmap(heightmap, show=False)
        
        start = time.time()
        action, _ = policy.predict(obs)
        time_elapsed_predict = time.time() - start
        obs, done, success, compactness, stability, time_elapsed_z = env.step_with_metrics(action)
        time_elapsed_list.append(time_elapsed_predict + time_elapsed_z)
        if success==0 or success==1:
            obj_packed += 1
        df_this_step = pd.DataFrame({'filename': filename, 'compactness_step': compactness, 'stability': stability, 'success': success, 'time_elapsed': time_elapsed_predict + time_elapsed_z}, index=[0])
        df_per_step = pd.concat([df_per_step, df_this_step], ignore_index=True)

    df_this_episode = pd.DataFrame({'filename': filename, 'obj_packed': obj_packed, 'compactness_episode': compactness, 'success_episode': success}, index=[0])
    df_results_episode = pd.concat([df_results_episode, df_this_episode], ignore_index=True)

print()
print('Mean time elapsed:', np.mean(time_elapsed_list))
print('Std time elapsed:', np.std(time_elapsed_list))

success_rate = df_results_episode[df_results_episode['success_episode'] == 1]['success_episode']
print('Success rate:', len(success_rate)/len(df_results_episode)*100)

df_results_episode.to_csv(os.path.join(tmp_path, 'test_results_episode.csv'))
df_per_step.to_csv(os.path.join(tmp_path, 'test_results_step.csv'))

# print success rate
print('Success rate:', len(df_results_episode[df_results_episode['success_episode'] == 1])/len(df_results_episode)*100)
# print mean compactness
print('Mean compactness:', df_results_episode['compactness_episode'].mean())
# print mean obj_packed
print('Mean obj_packed:', df_results_episode['obj_packed'].mean())
# print an empty line
print()
