from stable_baselines3.common.env_checker import check_env
from environment.physics0.simulator.packingGame import PackingGame
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from environment.physics0.model.custom_policy import ResnetCNN
import datetime
import os
import glob


EXPERIMENT_NAME = 'NEW_NAME'  # change this to your experiment name
CHECKPOINT = False  

# create a tmp_path for the experiment
tmp_path = "PATH/TO/tmp/FOLDER/" + EXPERIMENT_NAME
if tmp_path == "PATH/TO/tmp/FOLDER/" + EXPERIMENT_NAME:
    raise ValueError("Please change the tmp_path to your desired path. Edit the 'tmp_path' variable above.")

# check does the tmp_path exist
if os.path.exists(tmp_path):
    if not CHECKPOINT:
        raise ValueError(f"The path '{tmp_path}' already exists. Please choose another name for the experiment.")

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
BIN_SIZE = [0.345987, 0.227554, 0.1637639]
OBJECT_INFO = 'dataset/datas/object_info.npz'
ORDERED_OBJ = True
REWARD_FUNCTION = 'compactness'
ALPHA = None
UNPACKED_LIST_MIN = 0.7
UNPACKED_LIST_MAX = 0.9
CUSTOM_POLICY = None
# generate SEED from datetime
SEED = int(datetime.datetime.now().strftime("%d%H%M"))
set_random_seed(SEED)

train_env = make_vec_env(PackingGame, env_kwargs={'bin_size': BIN_SIZE,
                                                  'object_info': OBJECT_INFO,
                                                  'ordered_objs': ORDERED_OBJ, 
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

# Find latest checkpoint (modify this section)
if CHECKPOINT:
    checkpoints = glob.glob(f"{tmp_path}/sac_model_*.zip")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        # Load the model from checkpoint
        model = SAC.load(
            latest_checkpoint, 
            env=train_env,
            buffer_size=100000,
            verbose=1
        )
        model.set_logger(new_logger)
    else:
        raise ValueError(f"No checkpoints found. Please set CHECKPOINT to False to start a new training or ensure that checkpoints exist in the specified path: {tmp_path}.")

else:
    # Create new model without checkpoint
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
#                              best_model_save_path='PATH/TO/tmp/FOLDER/' + EXPERIMENT_NAME + '/eval_models',
#                              log_path='PATH/TO/tmp/FOLDER/' + EXPERIMENT_NAME + '/eval_logs',
#                              eval_freq=max(int(DESIRED_EVAL // N_ENVS), 1), 
#                              deterministic=True, 
#                              render=False,
#                              n_eval_episodes=10)


# callbacks
# callbacks = [checkpoint_callback, eval_callback]
callbacks = [checkpoint_callback]

# train for 10 milion steps with checkpointing every 1 million steps
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=500, callback=callbacks, progress_bar=True)

