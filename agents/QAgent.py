import stable_baselines3
from gym import spaces
from stable_baselines3 import SAC
import torch
import torch.nn
import numpy as np
import torch.optim as optim
import wandb


class QAgent(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden_layers=3, hidden_layer_size=50, gamma=0.9,
                 epsilon=0.3, lr=1e-4):
        super().__init__()
        self.gamma = gamma  # discount factor
        self.lr = lr
        self.epsilon = epsilon
        # self.replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 8)
        self.optimizer = optim.Adam
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size

    def convert_observation_space(self, obs_space):
        # not the best solution, but a way to convert the Observation Space from a Dict to a Box
        low = [obs_space[x].low[0] for x in obs_space]
        high = [obs_space[x].high[0] for x in obs_space]
        print(low, high, len(low))
        observation_space = spaces.Box(low=np.array(low), high=np.array(high),
                                       shape=(len(low),), dtype=np.float32)
        return observation_space

    def train_episode(self, env, verbose=0):
        MAXITERATIONS = 120
        state = torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0  # time step
        log_prob_actions = []
        rewards = []
        episode_reward = 0

        while True:
            action, states = self.agent.predict(state)
            state, reward, done, _ = env.step(action)
            if reward >= 1:
                if self.output_dim == 1:
                    print(
                        f"end position: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[0]) / action[0] * 100)[:5])}% difference)")
                elif self.output_dim == 2:
                    print(
                        f"end position Box 1: {state[0]}   \taction input: {action[0]}    \t{float(str((state[0] - action[0]) / action[0] * 100)[:5])}% difference)")
                    print(
                        f"end position Box 2: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[1]) / action[1] * 100)[:5])}% difference)")
            rewards.append(reward)
            episode_reward += reward
            t += 1
            reset = t == MAXITERATIONS
            # self.agent.observe(state, reward, done, reset)
            if done or reset:
                break
        try:  # todo: fix this and the loss calculation
            log_prob_actions = torch.cat(log_prob_actions)
        except RuntimeError:
            pass
        returns = self.calculate_returns(rewards, self.gamma)  # gamma: discount factor
        loss = self.update_policy(returns, log_prob_actions, self.optimizer)
        if verbose > 0:
            print(loss, episode_reward)
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
        if log_prob_actions == []:  # todo: fix
            return 0
        loss = - (returns * log_prob_actions).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, env):
        state = torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0  # time step
        done = False
        while not done:
            action, states = self.agent.predict(state)
            state, reward, done, _ = env.step(action)
            if reward >= 1:
                if self.output_dim == 1:
                    print(
                        f"end position: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[0]) / action[0] * 100)[:5])}% difference)")
                elif self.output_dim == 2:
                    print(
                        f"end position Box 1: {state[0]}   \taction input: {action[0]}    \t{float(str((state[0] - action[0]) / action[0] * 100)[:5])}% difference)")
                    print(
                        f"end position Box 2: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[1]) / action[1] * 100)[:5])}% difference)")
            R += reward
            t += 1
            reset = t == 200
            # self.agent.observe(state, reward, done, reset)
            if done or reset:
                break
        return R

    def train_loop(self, train_env, test_env, config, only_testing=False):
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery
        train_rewards = []
        test_rewards = []
        train_matches = 0  # count all perfectly balanced episodes
        test_matches = 0
        action_size = self.output_dim  # train_env.action_space.low.size

        if not only_testing:  # training & testing
            train_env.observation_space = self.convert_observation_space(train_env.observation_space)
            test_env.observation_space = self.convert_observation_space(test_env.observation_space)

            # self.replay_buffer = stable_baselines3.HerReplayBuffer(env=train_env, buffer_size=1000000, max_episode_length=1000)

            """self.agent = stable_baselines3.sac.SAC("MlpPolicy", train_env, learning_rate=0.0003, buffer_size=1000000, learning_starts=100,
                                     batch_size=256, tau=0.005, gamma=DISCOUNT_FACTOR, train_freq=1, gradient_steps=1,
                                     action_noise=None, replay_buffer_class=self.replay_buffer, replay_buffer_kwargs=None,
                                     optimize_memory_usage=False, ent_coef='auto', target_update_interval=1,
                                     target_entropy='auto', use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False,
                                     tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0,
                                     seed=None, device='auto', _init_setup_model=True)"""

            """self.agent = stable_baselines3.HER('MlpPolicy', train_env, SAC, n_sampled_goal=5,
                                          goal_selection_strategy='future',
                                          verbose=1, buffer_size=int(1e6),
                                          learning_rate=1e-3,
                                          gamma=0.95, batch_size=256,
                                          policy_kwargs=dict(layers=[256, 256, 256]))"""

            self.agent = stable_baselines3.sac.SAC('MlpPolicy', train_env).learn(5000)

        for episode in range(1, MAX_EPISODES + 1):
            if not only_testing:
                loss, train_reward = self.train_episode(train_env, verbose=0)
                if train_reward >= 1:  # todo: fix to another value
                    train_matches += 1
                train_rewards.append(train_reward)
                mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
                wandb.log({'train_reward': mean_train_rewards})
            test_reward = self.evaluate(test_env)
            if test_reward >= 1:
                test_matches += 1
            test_rewards.append(test_reward)
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            wandb.log({'test_rewards': mean_test_rewards})
            # todo: statistics
            """stats = self.agent.get_statistics()
            wandb.log({'average_q': stats[0][1]})
            wandb.log({'loss': stats[1][1]})"""
            if episode % PRINT_EVERY == 0:
                if not only_testing:
                    print(
                        f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                elif only_testing:
                    print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                # print(self.agent.get_statistics()) # todo
            if mean_test_rewards >= REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
        print()
        if not only_testing:
            train_success_rate = train_matches / MAX_EPISODES
            wandb.log({'train_success_rate': train_success_rate})
            print(
                f"Success rate of train episodes: {train_matches}/{MAX_EPISODES}={(train_matches / MAX_EPISODES) * 100:,.2f}%")
        print(
            f"Success rate of test episodes: {test_matches}/{MAX_EPISODES}={(test_matches / MAX_EPISODES * 100):,.2f}%")
        test_success_rate = test_matches / MAX_EPISODES
        wandb.log({'test_success_rate': test_success_rate})
        return train_rewards, test_rewards


if __name__ == '__main__':
    agent = QAgent(10, 10, 10)
