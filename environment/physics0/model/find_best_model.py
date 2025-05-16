from stable_baselines3 import SAC
from packingGame import PackingGame
import os
import json
import pandas as pd

MODEL_DIRECTORY = '/home/gojko/gojko-irbpp/tmp/sb3_log/0222_cs_09/'
TEST_FILE = 'test_set_list.json'

shortlist = [
'p28_s2',
'p58_s1',
'p52_s7',
'p77_s1',
'p24_s1',
'p6_s1' ,
'p54_s3',
'p19_s3',
'p47_s1',
'p11_s5',
'p88_s1',
'p49_s4',
'p36_s1',
'p5_s4',
'p47_s4',
'p55_s3',
'p41_s1',
'p53_s1',
'p54_s4',
'p75_s1',
'p16_s2',
'p49_s1',
'p67_s1',
'p80_s1',
'p5_s3',
'p71_s1',
'p81_s1',
]

df = pd.DataFrame(columns=['filename', 'reward', 'episode_length'])

# set the device to cuda 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# open json file
with open(TEST_FILE) as f:
    test_set_list = json.load(f)

env = PackingGame(ordered_objs=True, reward_function='compactness_stability', alpha=0.9, unpacked_list_min=0.7, unpacked_list_max=0.9)


for filename in os.listdir(MODEL_DIRECTORY):
    # check is the file .zip
    if filename.endswith('.zip'):

        # remove sac_model_ before the filename
        model_no = filename.split('sac_model_')[1]
        # remove _steps.zip from the filename
        model_no = model_no.split('_steps.zip')[0]

        # check if model_no is less than 8 million
        if int(model_no) < 7000000:
            # skip the model
            continue

        print('Loading model: ', filename)

        file = os.path.join(MODEL_DIRECTORY, filename)
        model = SAC.load(file)
        policy = model.policy.to('cuda')
        policy.eval()

        cumulative_reward_model = 0
        episode_length_model = 0

        for test_set in shortlist:
            
            unpacked_list = test_set_list[test_set]
            # print('Unpacked list: ', unpacked_list)

            obs, _ = env.reset(unpacked_list=unpacked_list)
            # print('Sorted objects: ', env.unpacked_list)

            done = False
            
            cumulative_reward_episode = 0
            episode_length_episode = 0  

            while not done:
                action, _ = policy.predict(obs)
                obs, reward, done, _, _ = env.step(action)

                cumulative_reward_episode += reward
                episode_length_episode += 1
            
            cumulative_reward_model += cumulative_reward_episode
            episode_length_model += episode_length_episode
        
        average_reward = cumulative_reward_model / len(shortlist)
        average_episode_length = episode_length_model / len(shortlist)

        df_temp = pd.DataFrame({'filename': [filename], 'reward': [average_reward], 'episode_length': [average_episode_length]})
        df = pd.concat([df, df_temp], ignore_index=True)

        print('Filename: ', filename)
        print('Reward: ', average_reward)
        print('Episode length: ', average_episode_length)

df.to_csv(os.path.join(MODEL_DIRECTORY, 'shortlist_results.csv'), index=False)

# sort by episode length
df.sort_values(by='episode_length', inplace=True)
print('Best models by episode length:', df.head(10))

# sort by reward
df.sort_values(by='reward', ascending=False, inplace=True)
print('Best models by reward:', df.head(10))