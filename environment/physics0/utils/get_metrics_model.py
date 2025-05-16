from packingGame import PackingGame, draw_heatmap
import os
import json
import pandas as pd
import time
import numpy as np
from stable_baselines3.sac.policies import CnnPolicy

model_path = 'policies/sac_model_11588400_steps.pkl'
tmp_path = 'tmp/sb3_log/0221'

# check does the tmp_path exist
if not os.path.exists(tmp_path):
    raise ValueError("The path does not exist. Please check the path.")
    exit()

TEST_FILE = 'test_set_list.json'
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
for filename in test_set_list:
    # if filename == 'p41_s1':
    #     break
    print(filename)
    unpacked_list = test_set_list[filename]
    obs, _ = env.reset(unpacked_list=unpacked_list)
    done = False
    obj_packed = 0

    while not done:
        heightmap = obs[0]
        draw_heatmap(heightmap)
        
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

print('Mean time elapsed:', np.mean(time_elapsed_list))
print('Std time elapsed:', np.std(time_elapsed_list))

success_rate = df_results_episode[df_results_episode['success_episode'] == 1]['success_episode']
print('Success rate:', len(success_rate)/len(df_results_episode)*100)

df_results_episode.to_csv(os.path.join(tmp_path, 'test_results_episode_cpu_1.csv'))
df_per_step.to_csv(os.path.join(tmp_path, 'test_results_step_cpuE_1.csv'))