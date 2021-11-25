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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.distributions as distributions

from environments.GymEnv import GymEnv
from ScaleEnvironment.Scale import Scale
from agents.VanillaGradMLP import VanillaGradMLP
import argparse

def create_envs(env_str, seed=42):
    if env_str == 'scale':
        train_env = Scale()
        test_env = Scale()
    else:
        train_env = GymEnv(env_str)
        train_env = train_env.create()
        test_env = GymEnv(env_str)
        test_env = test_env.create()
    train_env.seed(seed)
    test_env.seed(seed+1)
    return train_env, test_env

def get_env_dims(env):
    return env.observation_space.shape[0], env.action_space.n


def train_episode(env, policy, discount_factor):
    policy.train()
    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0
    state = env.reset()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_pred = policy(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample() # todo: counter --> only choose actions every x iterations
        log_prob_action = dist.log_prob(action)
        state, reward, done, _ = env.step(action.item())
        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_reward += reward
    log_prob_actions = torch.cat(log_prob_actions)
    returns = calculate_returns(rewards, discount_factor)
    loss = update_policy(returns, log_prob_actions, policy.optimizer)
    #print(loss,episode_reward)
    return loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def update_policy(returns, log_prob_actions, optimizer):
    returns = returns.detach()
    loss = - (returns * log_prob_actions).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(env, policy):
    policy.eval()
    done = False
    episode_reward = 0
    state = env.reset()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        state, reward, done, _ = env.step(action.item())
        episode_reward += reward
    return episode_reward

def train_loop(agent, config):
    MAX_EPISODES = config.episodes
    DISCOUNT_FACTOR = config.discount
    N_TRIALS = config.trials
    REWARD_THRESHOLD = config.threshold
    PRINT_EVERY = config.printevery
    train_rewards = []
    test_rewards = []
    for episode in range(1, MAX_EPISODES + 1):
        loss, train_reward = train_episode(train_env, agent, DISCOUNT_FACTOR)
        test_reward = evaluate(test_env, agent)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        if episode % PRINT_EVERY == 0:
            print(
                f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break
    return train_rewards, test_rewards


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
    input_dim, output_dim = 5, 3 #get_env_dims(train_env)
    agent = VanillaGradMLP(input_dim, 100, output_dim)
    mean_train_rewards, mean_test_rewards = train_loop(agent, args)
    plot_rewards(mean_train_rewards, mean_test_rewards, args.threshold)

