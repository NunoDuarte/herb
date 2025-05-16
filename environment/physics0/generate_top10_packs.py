from simulator.packingGame import PackingGame, unnormalize
from stable_baselines3.sac.policies import CnnPolicy
from simulator.space import draw_heatmap_norm
import os
import pandas as pd
import json

SAVE_IMAGES = True

PACKS_DIR = 'packs'

EXPERIMENT = 'shortlist'

MODEL_NAME = 'c09_s01'

MODEL_FILE = 'policies/0221_for_human_comparison.pkl'

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

shortlist = ['p11_s5']

# import test set list json
with open('test_set_list.json') as f:
    test_set_list = json.load(f)

env = PackingGame(visual=True, ordered_objs=True, reward_function='compactness_stability', alpha=0.6, unpacked_list_min=0.7, unpacked_list_max=0.9)
    
policy = CnnPolicy.load(MODEL_FILE)
policy.eval()

for test_set in shortlist:

    pack_name_full = MODEL_NAME + test_set
    export_dir = os.path.join(PACKS_DIR, EXPERIMENT, MODEL_NAME, pack_name_full)
    

    # create the export directory if it does not exist
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # get the unpacked list
    unpacked_list = test_set_list[test_set]

    print(unpacked_list)

    

    df = pd.DataFrame(columns=['objectName', 'x', 'y', 'theta'])

    terminated = False

    obs, _ = env.reset(unpacked_list=unpacked_list)
    # heightmap = obs[0]
    # draw_heatmap_norm(heightmap)

    i = 0
    while not terminated:

        if SAVE_IMAGES:
            env.render(heightname=os.path.join(export_dir, pack_name_full + '_height_' + f'{i}.png'), colorname=os.path.join(export_dir, pack_name_full + '_color_' + f'{i}.png'))

        action, _states = policy.predict(obs, deterministic=True)

        objname = env.unpacked_list[0]

        x = unnormalize(action[0], 0.0, 0.345987)
        y = unnormalize(action[1], 0.0, 0.227554)
        theta = unnormalize(action[2], 0.0, 180.0)

        # append to the dataframe
        df = df._append({'objectName': objname, 'x': x, 'y': y, 'theta': theta}, ignore_index=True)

        print('X:', unnormalize(action[0], 0.0, 0.345987), 'Y:', unnormalize(action[1], 0.0, 0.227554), 'Theta:', unnormalize(action[2], 0.0, 180.0)) 
        obs, rewards, terminated, truncated, info = env.step(action)
        # remove first dimension of obs
        heightmap = obs[0]
        # draw_heatmap_norm(heightmap)
        # print(rewards)

        i += 1

        if terminated:
            # export the dataframe as a csv
            df.to_csv(os.path.join(export_dir, pack_name_full + '.csv'))

            if SAVE_IMAGES:
                env.render(heightname=os.path.join(export_dir, pack_name_full + '_height_' + 'final.png'), colorname=os.path.join(export_dir, pack_name_full + '_color_' + 'final.png'))

        