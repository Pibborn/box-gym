import time
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
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from environments.GymEnv import GymEnv
from ScaleEnvironment.Scale import Scale
from ScaleEnvironment.ScaleExperiment import ScaleExperiment
from agents.VanillaGradMLP import VanillaGradMLP
import argparse

TRAINING = 1
TESTING = 2

def create_envs(env_str, seed=42, do_render=True, randomness=False):
    if env_str == 'scale':
        train_env = Scale(rendering=do_render)
        test_env = Scale(rendering=do_render)
    elif env_str == 'scale_exp':
        train_env = ScaleExperiment(rendering=do_render, randomness=randomness)
        test_env = ScaleExperiment(rendering=do_render, randomness=randomness)
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

def plot_test_rewards(test_rewards, threshold):
    plt.figure(figsize=(12,8))
    #plt.title(f"Randomness = {randomness}, discount = {discount}, dropout = {dropout}", fontdict={'fontsize': 20})
    plt.plot(test_rewards, label='Test Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # 1: run the agent and train and test him, save the agent in a file
    # 2: load the agent from the file and only test him
    mode = TRAINING
    randomness = False
    rendering = False
    if mode == TESTING:
        only_testing = True
    else:
        only_testing = False
    discount = 0.99  # old default: 0.99
    dropout = 0.2    # old default: 0.2

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('envname')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=10000) # old default: 1000
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--printevery', type=int, default=10)
    parser.add_argument('--discount', type=float, default=discount) # old default: 0.99
    parser.add_argument('--threshold', type=float, default=2000) # old default: 475
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    if mode == 1:  # train + test new agent
        train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=rendering, randomness=randomness)  # do_render=True
        input_dim, output_dim = get_env_dims(train_env)
        agent = VanillaGradMLP(input_dim, 100, output_dim, dropout=dropout, uses_scale=args.envname=='scale',
                           scale_exp=args.envname=='scale_exp')
        mean_train_rewards, mean_test_rewards = agent.train_loop(train_env, test_env, args, only_testing=only_testing)
        # save the trained agent
        with open('agent', 'wb') as agent_file:
            pickle.dump(agent, agent_file)
        plot_rewards(mean_train_rewards, mean_test_rewards, args.threshold)

    else:  # load old agent and test him
        # load the agent
        with open('agent', 'rb') as agent_file:
            agent = pickle.load(agent_file)
            # use the loaded agent
            train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=rendering, randomness=randomness)
            _, mean_test_rewards = agent.train_loop(train_env, test_env, args, only_testing=only_testing)
        plot_test_rewards(mean_test_rewards, args.threshold)

    end_time = time.time()
    print(end_time - start_time)
