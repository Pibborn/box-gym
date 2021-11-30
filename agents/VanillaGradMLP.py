from agents.initializers import xavier_init
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.distributions as distributions
import numpy as np


class VanillaGradMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5,
                 lr=0.05):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

    def init_weights(self):
        self.apply(xavier_init)

    def train_episode(self, env, discount_factor):
        self.train()
        log_prob_actions = []
        rewards = []
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_pred = self(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()  # todo: counter --> only choose actions every x iterations
            log_prob_action = dist.log_prob(action)
            state, reward, done, _ = env.step(action.item())
            log_prob_actions.append(log_prob_action)
            rewards.append(reward)
            episode_reward += reward
        log_prob_actions = torch.cat(log_prob_actions)
        returns = self.calculate_returns(rewards, discount_factor)
        loss = self.update_policy(returns, log_prob_actions, self.optimizer)
        # print(loss,episode_reward)
        return loss, episode_reward

    def calculate_returns(self, rewards, discount_factor, normalize=True):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        return returns

    def update_policy(self, returns, log_prob_actions, optimizer):
        returns = returns.detach()
        loss = - (returns * log_prob_actions).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, env, policy):
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

    def train_loop(self, train_env, test_env, config):
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery
        train_rewards = []
        test_rewards = []
        for episode in range(1, MAX_EPISODES + 1):
            loss, train_reward = self.train_episode(train_env, DISCOUNT_FACTOR)
            test_reward = self.evaluate(test_env)
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
