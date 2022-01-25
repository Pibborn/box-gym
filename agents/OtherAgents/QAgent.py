import pfrl
import torch
import torch.nn
import numpy as np
import torch.optim as optim
import wandb
from agents.AgentInterface import Agent


class QAgent(Agent):
    def __init__(self, input_dim, output_dim, num_hidden_layers=3, hidden_layer_size=50, gamma=0.9,
                 epsilon=0.3, lr=1e-4):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.gamma = gamma  # discount factor
        self.lr = lr
        self.epsilon = epsilon
        self.replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 8)
        self.optimizer = optim.Adam
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size

    def train_episode(self, env, verbose=1):
        MAXITERATIONS = 120
        state = torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0  # time step
        log_prob_actions = []
        rewards = []
        episode_reward = 0

        while True:
            action = self.agent.act(state)
            state, reward, done, _ = env.step(action)
            if reward >= 1:
                print(
                    f"end position: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[0]) / action[0] * 100)[:5])}% difference")
            rewards.append(reward)
            episode_reward += reward
            t += 1
            reset = t == MAXITERATIONS
            self.agent.observe(state, reward, done, reset)
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

    def evaluate(self, env):
        with self.agent.eval_mode():
            state = torch.tensor(env.reset())
            R = 0  # return (sum of rewards)
            t = 0  # time step
            done = False
            while not done:
                action = self.agent.act(state)
                #print(action)
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
                self.agent.observe(state, reward, done, reset)
                if done or reset:
                    break
            return R

    def train_loop(self, train_env, test_env, config, verbose=1, sde=False, only_testing=False):
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
            q_func = pfrl.q_functions.FCQuadraticStateQFunction(
                self.input_dim,
                action_size,
                n_hidden_channels=self.input_dim,
                n_hidden_layers=self.n_hidden_layers,
                action_space=train_env.action_space,
            )
            wandb.watch(q_func)
            # Use the Ornstein-Uhlenbeck process for exploration
            ou_sigma = (train_env.action_space.high - train_env.action_space.low) * 0.2

            # self.explorer = pfrl.explorers.AdditiveOU(sigma=ou_sigma)
            self.explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=self.epsilon,
                                                                 random_action_func=train_env.action_space.sample)

            self.optimizer = self.optimizer(q_func.parameters())
            gpu = -1  # 1: use gpu&cpu, -1: only use cpu
            self.agent = pfrl.agents.DQN(  # pfrl.agents.DQN(
                q_func,
                self.optimizer,
                self.replay_buffer,
                gamma=DISCOUNT_FACTOR,  # self.gamma
                explorer=self.explorer,
                replay_start_size=128,
                update_interval=1,
                target_update_interval=1,
                minibatch_size=32,
                gpu=gpu
            )

        for episode in range(1, MAX_EPISODES + 1):
            if not only_testing:
                loss, train_reward = self.train_episode(train_env, verbose=0)
                if train_reward >= 1:   #todo: fix to another value
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
            stats = self.agent.get_statistics()
            wandb.log({'average_q': stats[0][1]})
            wandb.log({'loss': stats[1][1]})
            if episode % PRINT_EVERY == 0:
                if not only_testing:
                    print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                elif only_testing:
                    print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                print(self.agent.get_statistics())
            if mean_test_rewards >= REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
        print()
        if not only_testing:
            train_success_rate = train_matches / MAX_EPISODES
            wandb.log({'train_success_rate': train_success_rate})
            print(f"Success rate of train episodes: {train_matches}/{MAX_EPISODES}={(train_matches / MAX_EPISODES) * 100:,.2f}%")
        print(f"Success rate of test episodes: {test_matches}/{MAX_EPISODES}={(test_matches / MAX_EPISODES * 100):,.2f}%")
        test_success_rate = test_matches / MAX_EPISODES
        wandb.log({'test_success_rate': test_success_rate})
        return train_rewards, test_rewards

    def save_agent(self, location): # todo
        pass

    def load_agent(self, location):
        pass