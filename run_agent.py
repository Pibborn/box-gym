import gym
import math
import random
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'WebAgg'
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from environments.GymEnv import GymEnv
from ScaleEnvironment.Scale import Scale
from agents.VanillaGradMLP import VanillaGradMLP
import argparse

def create_envs(env_str, seed=42):
    if env_str == 'scale':
        train_env = Scale(rendering=True)
        test_env = Scale(rendering=True)
    else:
        train_env = GymEnv(env_str)
        train_env = train_env.create()
        test_env = GymEnv(env_str)
        test_env = test_env.create()
    train_env.seed(seed)
    test_env.seed(seed+1)
    return train_env, test_env

def get_env_dims(env):
    try:
        return env.observation_space.shape[0], env.action_space.n
    except TypeError: # if observation_space and action_space are dicts
        return len(env.observation_space), len(env.action_space)

def plot_rewards(train_rewards, test_rewards, threshold):
    plt.figure(figsize=(12,8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('envname')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--printevery', type=int, default=10)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--threshold', type=float, default=475)
    args = parser.parse_args()
    train_env, test_env = create_envs(args.envname, seed=args.seed)
    input_dim, output_dim = get_env_dims(train_env)
    agent = VanillaGradMLP(input_dim, 100, output_dim, dropout=0.2, uses_scale=args.envname=='scale')
    mean_train_rewards, mean_test_rewards = agent.train_loop(train_env, test_env, args)
    plot_rewards(mean_train_rewards, mean_test_rewards, args.threshold)

