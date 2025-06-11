from stable_baselines3.common.env_checker import check_env
from packingGame import PackingGame
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from custom_policy import ResnetCNN
import datetime
import os


EXPERIMENT_NAME = '0228_CO'

tmp_path = "/mnt/home/nuno/code/github/archive/gojko-irbpp/tmp/sb3_log/" + EXPERIMENT_NAME

# check does the tmp_path exist
if os.path.exists(tmp_path):
    raise ValueError("The path already exists. Please choose another name for the experiment.")
    exit()

new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

# set cuda device 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# def make_env(rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """

#     def _init():
#         env = PackingGame()
#         # use a seed for reproducibility
#         # Important: use a different seed for each environment
#         # otherwise they would generate the same experiences
#         env.reset(seed=seed + rank)
#         return env

#     set_random_seed(seed)
#     return _init

# train_env = PackingGame()

N_ENVS = 100
DESIRED_CP = 200000
DESIRED_EVAL = 500
TOTAL_TIMESTEPS = 10000000
ORDERED_OBJ = True
REWARD_FUNCTION = 'compactness'
ALPHA = None
UNPACKED_LIST_MIN = 0.7
UNPACKED_LIST_MAX = 0.9
CUSTOM_POLICY = None
# generate SEED from datetime
SEED = int(datetime.datetime.now().strftime("%d%H%M"))
set_random_seed(SEED)

train_env = make_vec_env(PackingGame, env_kwargs={'ordered_objs': ORDERED_OBJ, 
                                                  'reward_function': REWARD_FUNCTION, 
                                                  'alpha': ALPHA,
                                                  'unpacked_list_min': UNPACKED_LIST_MIN,
                                                  'unpacked_list_max': UNPACKED_LIST_MAX},
                                                  n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': 'fork'},
                                                  seed=SEED,
                                        )
# train_env = SubprocVecEnv([lambda: PackingGame() for i in range(4)], start_method='fork')

# eval_env = PackingGame()
# wrap the environment in the monitor wrapper
# eval_env = Monitor(eval_env, tmp_path)

if CUSTOM_POLICY == None:
  model = SAC('CnnPolicy', train_env, verbose=1, buffer_size=100000)
elif CUSTOM_POLICY == 'RESNET':
  policy_kwargs = dict(
    features_extractor_class=ResnetCNN,
    features_extractor_kwargs=dict(features_dim=1024),
  )
  model = SAC('CnnPolicy', train_env, policy_kwargs=policy_kwargs, verbose=1)
else:
  raise ValueError("CUSTOM_POLICY must be None or 'RESNET'")

model.set_logger(new_logger)

# Save a checkpoint every DESIRED_CP steps
checkpoint_callback = CheckpointCallback(
  save_freq= max(int(DESIRED_CP // N_ENVS), 1),
  save_path=tmp_path,
  name_prefix="sac_model",
  save_replay_buffer=False,
  save_vecnormalize=False,
)

#save hyperparameters
hyperparameters = {
    'order' : ORDERED_OBJ,
    'reward_function' : REWARD_FUNCTION,
    'alpha' : ALPHA,
    'unpacked_list_min' : UNPACKED_LIST_MIN,
    'unpacked_list_max' : UNPACKED_LIST_MAX,
    'custom_policy' : CUSTOM_POLICY,
    'seed' : SEED
}
with open(f'{tmp_path}/hyperparameters.txt', 'w') as f:
    f.write(str(hyperparameters))


# Eval callback
# eval_callback = EvalCallback(eval_env, 
#                              best_model_save_path='/home/gojko/gojko-irbpp/tmp/sb3_log/' + EXPERIMENT_NAME + '/best_model', 
#                              log_path='/home/gojko/gojko-irbpp/tmp/sb3_log/' + EXPERIMENT_NAME + '/eval_logs', 
#                              eval_freq=max(int(DESIRED_EVAL // N_ENVS), 1), 
#                              deterministic=True, 
#                              render=False,
#                              n_eval_episodes=10)


# callbacks
# callbacks = [checkpoint_callback, eval_callback]
callbacks = [checkpoint_callback]

# train for 10 milion steps with checkpointing every 1 million steps
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=500, callback=callbacks, progress_bar=True)

