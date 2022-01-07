import math

from agents.initializers import xavier_init
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from torch import autograd
import numpy as np
from collections import OrderedDict
from ScaleEnvironment.ScaleExperiment import rescale_movement

class VanillaGradMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5,
                 lr=0.0005, uses_scale=True, scale_exp=False):
        super().__init__()
        self.uses_scale = uses_scale
        self.scale_exp = scale_exp
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        if not uses_scale:
            self.fc_2 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc_2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        with autograd.detect_anomaly():
            x = self.fc_1(x)
            x = self.dropout(x)
            x = F.sigmoid(x)
            x = self.fc_2(x)
        return x

    def init_weights(self):
        self.apply(xavier_init)

    def train_episode(self, env, discount_factor, verbose=0):
        MAXITERATIONS = 120
        self.train()
        log_prob_actions = []
        rewards = []
        done = False
        episode_reward = 0
        state = env.reset()
        for _ in range(MAXITERATIONS):
            if done:
                break
            state = torch.FloatTensor(state).unsqueeze(0)
            action_pred = self(state)
            if not self.uses_scale and not self.scale_exp:
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                action = dist.sample()
                log_prob_action = dist.log_prob(action)
                log_prob_actions.append(log_prob_action)
                action = action.item()
            elif self.scale_exp:
                box1_pos = torch.tanh(action_pred[0][0])
                box2_pos = torch.tanh(action_pred[0][1])
                box1_pos = rescale_movement((-1, 1), box1_pos)
                box2_pos = rescale_movement((-1, 1), box2_pos)
                dist_box1 = distributions.Normal(torch.reshape(box1_pos, (1, 1)), 0.2)
                dist_box2 = distributions.Normal(torch.reshape(box2_pos, (1, 1)), 0.2)
                box1_action = dist_box1.sample()
                box2_action = dist_box2.sample()
                box1_prediction = np.clip(box1_action, -13, -2)  # only allow values within the given action_space
                box2_prediction = np.clip(box2_action, 2, 13)
                action = np.array([box1_prediction.item(), box2_prediction.item()])
                log_prob_actions.append(dist_box1.log_prob(box1_action) + dist_box2.log_prob(box2_action))
                #log_prob_actions.append(dist_box1.log_prob(box1_prediction) + dist_box2.log_prob(box2_prediction))
            else:
                which_box = torch.sigmoid(action_pred[0][0])
                movement = torch.tanh(action_pred[0][1])
                dist_box = distributions.Categorical(torch.reshape(which_box, (1, 1)))
                dist_movement = distributions.Normal(torch.reshape(movement, (1, 1)), 0.2)
                box_action = dist_box.sample()
                movement_action = dist_movement.sample()
                action = OrderedDict([('box', np.array([[box_action.item()]])), ('delta_pos', np.array([[movement_action.item()]]))])
                log_prob_actions.append(dist_box.log_prob(box_action) + dist_movement.log_prob(movement_action))
            state, reward, done, _ = env.step(action)
            if reward >= 1: #todo: fix --> should be for both boxes
                print(
                    f"end position Box 1: {state[0]}   \taction input: {action[0]}    \t{float(str((state[0] - action[0]) / action[0] * 100)[:5])}% difference)")
                print(
                    f"end position Box 2: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[1]) / action[1] * 100)[:5])}% difference)")
            rewards.append(reward)
            episode_reward += reward
        try:
            log_prob_actions = torch.cat(log_prob_actions)
        except RuntimeError:
            pass
        returns = self.calculate_returns(rewards, discount_factor)
        loss = self.update_policy(returns, log_prob_actions, self.optimizer)
        if verbose > 0:
            print(loss,episode_reward)
        return loss, episode_reward

    def calculate_returns(self, rewards, discount_factor, normalize=True):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalize:
            if returns.size()[0] > 1:
                returns = (returns - returns.mean()) / returns.std()
            else:
                pass
        return returns

    def update_policy(self, returns, log_prob_actions, optimizer):
        returns = returns.detach()
        loss = - (returns * log_prob_actions).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, env):
        self.eval()
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred = self(state)
                if not self.uses_scale and not self.scale_exp:
                    action_prob = F.softmax(action_pred, dim=-1)
                    action = torch.argmax(action_prob, dim=-1)
                    action = action.item()
                elif self.scale_exp:
                    box1_pos = torch.tanh(action_pred[0][0])
                    box2_pos = torch.tanh(action_pred[0][1])
                    box1_pos = rescale_movement((-1, 1), box1_pos)
                    box2_pos = rescale_movement((-1, 1), box2_pos)
                    dist_box1 = distributions.Normal(torch.reshape(box1_pos, (1, 1)), 0.2)
                    dist_box2 = distributions.Normal(torch.reshape(box2_pos, (1, 1)), 0.2)
                    box1_action = dist_box1.sample()
                    box2_action = dist_box2.sample()
                    box1_prediction = np.clip(box1_action, -13, -2) # only allow values within the given action_space
                    box2_prediction = np.clip(box2_action, 2, 13)
                    action = np.array([box1_prediction.item(), box2_prediction.item()])
                else:
                    which_box = torch.sigmoid(action_pred[0][0])
                    movement = torch.tanh(action_pred[0][1])
                    dist_box = distributions.Categorical(torch.reshape(which_box, (1, 1)))
                    dist_movement = distributions.Normal(torch.reshape(movement, (1, 1)), 0.2)
                    box_action = dist_box.sample()
                    movement_action = dist_movement.sample()
                    action = OrderedDict(
                        [('box', np.array([[box_action]])), ('delta_pos', np.array([[movement_action]]))]
                    )
            state, reward, done, _ = env.step(action)
            if reward >= 1:
                print(
                    f"end position Box 1: {state[0]}   \taction input: {action[0]}    \t{float(str((state[0] - action[0]) / action[0] * 100)[:5])}% difference)")
                print(
                    f"end position Box 2: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[1]) / action[1] * 100)[:5])}% difference)")
            episode_reward += reward
        return episode_reward

    def train_loop(self, train_env, test_env, config, only_testing = False):
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery
        train_rewards = []
        test_rewards = []
        train_matches = 0
        test_matches = 0
        for episode in range(1, MAX_EPISODES + 1):
            if not only_testing:
                loss, train_reward = self.train_episode(train_env, DISCOUNT_FACTOR, verbose=0)
                test_reward = self.evaluate(test_env)
                train_rewards.append(train_reward)
                test_rewards.append(test_reward)
                if train_reward >= 1:
                    train_matches += 1
                if test_reward >= 1:
                    test_matches += 1
                mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                if episode % PRINT_EVERY == 0:
                    print(
                        f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                if mean_test_rewards >= REWARD_THRESHOLD:
                    print(f'Reached reward threshold in {episode} episodes')
                    break
            else:
                test_reward = self.evaluate(test_env)
                test_rewards.append(test_reward)
                if test_reward >= 1:
                    test_matches += 1
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                if episode % PRINT_EVERY == 0:
                    print(
                        f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                if mean_test_rewards >= REWARD_THRESHOLD:
                    print(f'Reached reward threshold in {episode} episodes')
                    break
        print()
        if not only_testing:
            print(
                f"Success rate of train episodes: {train_matches}/{MAX_EPISODES}={(train_matches / MAX_EPISODES) * 100:,.2f}%")
        print(
            f"Success rate of test episodes: {test_matches}/{MAX_EPISODES}={(test_matches / MAX_EPISODES * 100):,.2f}%")
        return train_rewards, test_rewards
