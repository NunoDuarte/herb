import numpy as np
from simulator.space import draw_heatmap_norm, draw_heatmap
from simulator.packingGame import PackingGame
from ros.baxter_demos import prepare_obs_camera
from stable_baselines3.sac.policies import CnnPolicy


# unnormalize from [-1 to 1]
def unnormalize(x, min, max):
    return 0.5 * (x + 1) * (max - min) + min

env = PackingGame(visual=True, ordered_objs=True)
env.reset()

model_pth = 'policies/0221_for_human_comparison.pkl'
policy = CnnPolicy.load(model_pth)
policy.eval()
print('Model loaded')

image = np.load('environment/physics0/images/heightmap.npy')
image = image / 1000.0
print(np.min(image))
print(np.max(image))
obs = prepare_obs_camera(image, '003 Cracker Box', camera_offset=0.0)

draw_heatmap_norm(obs[0])

action, _ = policy.predict(obs)
action_simulated = action.copy()
print(action)

env.step(action_simulated, '003 Cracker Box')

print('x: ', action[0], 'y: ', action[1], 'theta: ', action[2])

x = unnormalize(action[0], 0.0, 0.345987)
y = unnormalize(action[1], 0.0, 0.227554)
theta = unnormalize(action[2], 0.0, 180.0)

# print the action
print('X:', x, 'Y:', y, 'Theta:', theta)






