from stable_baselines3 import SAC
from environment.physics0.simulator.packingGame import PackingGame

# to export your trained model to .pkl format
file = '/PATH/TO/zip/FILE'
if file == '/PATH/TO/zip/FILE':
    raise ValueError("Please change the name of your file to your desired. Edit the 'file' variable above.")
    
model = SAC.load(file)

# save the policy
policy = model.policy
policy.save('policies/new_model.pkl')
