from simulator.packingGame import PackingGame, unnormalize
from stable_baselines3.sac.policies import CnnPolicy
from simulator.space import draw_heatmap_norm

env = PackingGame(visual=True, ordered_objs=True, reward_function='compactness_stability', alpha=0.6)
file = 'policies/sac_model_c09s01.pkl'
policy = CnnPolicy.load(file)

terminated = False
unpacked_list = ["017 Orange",'014 Lemon', '007 Tuna Fish Can', '005 Tomato Coup Can', '004 Sugar Box']
obs, _ = env.reset(unpacked_list=unpacked_list)

while True:
    action, _states = policy.predict(obs, deterministic=False)
    print('X:', unnormalize(action[0], 0.0, 0.345987), 'Y:', unnormalize(action[1], 0.0, 0.227554), 'Theta:', unnormalize(action[2], 0.0, 180.0)) 
    obs, rewards, terminated, truncated, info = env.step(action)
    # remove first dimension of obs
    heightmap = obs[0]
    draw_heatmap_norm(heightmap)
    print(rewards)
    if rewards == -1 and terminated:
        print('Bad packing. Reset!')
        obs, _ = env.reset(unpacked_list=unpacked_list)
    elif rewards != -1 and terminated:
        print('Finished Packing. Done!')
        exit(1)
        

