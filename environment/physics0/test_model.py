from simulator.packingGame import PackingGame, unnormalize
from stable_baselines3.sac.policies import CnnPolicy
from simulator.space import draw_heatmap_norm
import os
import pandas as pd
import json

SAVE_IMAGES = False

PACKS_DIR = 'packs'

PACK_NAME = 'pack0'

MODEL_FILE = 'policies/0221_for_human_comparison.pkl'

export_dir = os.path.join(PACKS_DIR, PACK_NAME)

# create the export directory if it does not exist
if not os.path.exists(export_dir):
    os.makedirs(export_dir)


env = PackingGame(visual=True, ordered_objs=True, reward_function='compactness', unpacked_list_min=0.7, unpacked_list_max=0.9)
policy = CnnPolicy.load(MODEL_FILE)

df = pd.DataFrame(columns=['objectName', 'x', 'y', 'theta'])

terminated = False
unpacked_list = ["017 Orange",
                 '014 Lemon', 
                 '007 Tuna Fish Can', 
                 '005 Tomato Coup Can', 
                 '004 Sugar Box',
                 "003 Cracker Box",
                 "008 Pudding Box",
                 "006 Mustard Bottle",
                 "010 Potted Meat Can",
                 "025 Mug",
                 "016 Pear",
                 "012 Strawberry",
                 "011 Banana",
                 "057 Racquetball"
                 ]

obs, _ = env.reset(unpacked_list=unpacked_list)
heightmap = obs[0]
draw_heatmap_norm(heightmap)

i = 0
while not terminated:

    if SAVE_IMAGES:
        env.render(heightname=os.path.join(export_dir, PACK_NAME + '_height_' + f'{i}.png'), colorname=os.path.join(export_dir, PACK_NAME + '_color_' + f'{i}.png'))

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
    draw_heatmap_norm(heightmap)
    print(rewards)

    i += 1

    if terminated:
        # export the dataframe as a csv
        df.to_csv(os.path.join(export_dir, PACK_NAME + '.csv'))

        if SAVE_IMAGES:
            env.render(heightname=os.path.join(export_dir, PACK_NAME + '_height_' + 'final.png'), colorname=os.path.join(export_dir, PACK_NAME + '_color_' + 'final.png'))

        
