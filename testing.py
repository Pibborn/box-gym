from xvfbwrapper import Xvfb
vdisplay = Xvfb()
vdisplay.start()
import numpy as np
from gym import spaces
import gym
from stable_baselines3 import PPO, A2C, SAC, DQN, DDPG  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb
import argparse

# arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('envname')
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--entity', type=str, default='jgu-wandb')
parser.add_argument('--random', action='store_true')
args = parser.parse_args()

# wandb init
wandb.init(config=args, project="box-gym", entity=args.entity, sync_tensorboard=True)

# Instantiate the env
from ScaleEnvironment.ScaleExperiment import ScaleExperiment

env = ScaleExperiment(rendering=False, randomness=args.random, actions=2)
low = [env.observation_space[x].low[0] for x in env.observation_space]
high = [env.observation_space[x].high[0] for x in env.observation_space]
env.observation_space = spaces.Box(low=np.array(low), high=np.array(high),
                                   shape=(len(low),), dtype=np.float32)

# wrap it
env = DummyVecEnv([lambda: env])
env = VecVideoRecorder(env, 'videos', record_video_trigger=lambda x: x % 1000 == 0, video_length=200)

# Train the agentTe
wandb_callback = WandbCallback(gradient_save_freq=100,
                               model_save_path="results/temp",
                               verbose=0)
model = SAC('MlpPolicy', env, verbose=0, use_sde=False, tensorboard_log='results/temp')
model.learn(args.steps, callback=wandb_callback)
model.save('SAC_Model2')
model = SAC.load('SAC_Model2', env=env)

#model = A2C('MlpPolicy', env, verbose=1, use_sde=True).learn(5000)
#model.save('A2C_Model')
#model = A2C.load('A2C_Model', env=env)


# Test the trained agent
obs = env.reset()
n_steps = 20
episodes = 500
success = 0

#evaluate_policy(model, env, n_eval_episodes=10, render=True)

obs = env.reset()
env = ScaleExperiment(rendering=True, randomness=args.random, actions=2)
while True:
    for _ in range(100):
        action, states = model.predict(obs)
        state, reward, done, _ = env.step(action)
        if done:
            break
