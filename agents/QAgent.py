import pfrl
import torch
import torch.nn
import numpy as np
import torch.optim as optim


class QAgent():

    def __init__(self, input_dim, output_dim, num_hidden_layers=3, hidden_layer_size=50, gamma=0.9,
                 epsilon=0.3, lr=1e-4):
        super().__init__()
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
        self.optimizer = optim.Adam
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size

    def train_episode(self, env, verbose=0):
        max_episode_len = 20
        state = torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while True:
            action = self.agent.act(state)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
            reset = t == max_episode_len
            self.agent.observe(obs, reward, done, reset)
            if done or reset:
                break
        return 0, R # TODO: loss calc

    def evaluate(self, env):
        with self.agent.eval_mode():
            obs = torch.tensor(env.reset())
            R = 0
            t = 0
            while True:
                action = self.agent.act(obs)
                obs, r, done, _ = env.step(action)
                R += r
                t += 1
                reset = t == 200
                self.agent.observe(obs, r, done, reset)
                if done or reset:
                    break
            return R

    def train_loop2(self, train_env, test_env, config, only_testing=False):
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery
        train_rewards = []
        test_rewards = []
        action_size = 2  # train_env.action_space.low.size

        if not only_testing:  # training & testing
            q_func = pfrl.q_functions.FCQuadraticStateQFunction(
                self.input_dim,
                action_size,
                n_hidden_channels=self.input_dim,
                n_hidden_layers=self.n_hidden_layers,
                action_space=train_env.action_space,
            )
            # Use the Ornstein-Uhlenbeck process for exploration
            ou_sigma = (train_env.action_space.high - train_env.action_space.low) * 0.2
            self.explorer = pfrl.explorers.AdditiveOU(sigma=ou_sigma)
            self.optimizer = self.optimizer(q_func.parameters())
            self.agent = pfrl.agents.DQN(
                q_func,
                self.optimizer,
                self.replay_buffer,
                gamma=self.gamma,
                explorer=self.explorer,
                replay_start_size=500,
                update_interval=1,
                target_update_interval=1,
                minibatch_size=32
            )

            for episode in range(1, MAX_EPISODES + 1):
                loss, train_reward = self.train_episode(train_env, verbose=0)
                test_reward = self.evaluate(test_env)
                train_rewards.append(train_reward)
                test_rewards.append(test_reward)
                mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                if episode % PRINT_EVERY == 0:
                    print(
                        f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                    print(self.agent.get_statistics())
                if mean_test_rewards >= REWARD_THRESHOLD:
                    print(f'Reached reward threshold in {episode} episodes')
                    break

        else:  # only testing
            for episode in range(1, MAX_EPISODES + 1):
                test_reward = self.evaluate(test_env)
                test_rewards.append(test_reward)
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                if episode % PRINT_EVERY == 0:
                    print(
                        f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                if mean_test_rewards >= REWARD_THRESHOLD:
                    print(f'Reached reward threshold in {episode} episodes')
                    break
            return [], test_rewards
        return train_rewards, test_rewards

    def train_loop(self, train_env, test_env, config, only_testing=False): # new version
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery
        train_rewards = []
        test_rewards = []
        action_size = 2 #train_env.action_space.low.size

        if not only_testing: # training & testing
            # Use NAF to apply DQN to continuous action spaces
            q_func = pfrl.q_functions.FCQuadraticStateQFunction(
                self.input_dim,
                action_size,
                n_hidden_channels=self.input_dim,
                n_hidden_layers=self.n_hidden_layers,
                action_space=train_env.action_space,
            )
            # Use the Ornstein-Uhlenbeck process for exploration
            ou_sigma = (train_env.action_space.high - train_env.action_space.low) * 0.2
            self.explorer = pfrl.explorers.AdditiveOU(sigma=ou_sigma)
            self.optimizer = self.optimizer(q_func.parameters())
            self.agent = pfrl.agents.DQN(
                q_func,
                self.optimizer,
                self.replay_buffer,
                gamma=self.gamma,
                explorer=self.explorer,
                replay_start_size=500,
                update_interval=1,
                target_update_interval=1,
                minibatch_size=32
            )

        else:  # only testing
            # use the agent save in the object
            pass

        agent, history = pfrl.experiments.train_agent_with_evaluation(
            agent=self.agent,
            env=train_env,
            steps=MAX_EPISODES,
            eval_n_steps=None,
            eval_n_episodes=N_TRIALS,
            eval_interval=100,
            outdir='results/',
            eval_env=test_env,
            train_max_episode_len=120,
            eval_during_episode=True,
        )
        eval_rewards = [h['eval_score'] for h in history]
        print(eval_rewards)

        #for episode in range(1, MAX_EPISODES + 1):
        #    loss, train_reward = self.train_episode(train_env, verbose=0)
        #    test_reward = self.evaluate(test_env)
        #    train_rewards.append(train_reward)
        #    test_rewards.append(test_reward)
        #    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        #    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        #    if episode % PRINT_EVERY == 0:
        #        print(
        #            f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        #        print(self.agent.get_statistics())
        #    if mean_test_rewards >= REWARD_THRESHOLD:
        #        print(f'Reached reward threshold in {episode} episodes')
        #        break
        #return train_rewards, test_rewards

        # convert rewards to tuple form (x,y)
        test_rewards = [(h['cumulative_steps'], h['eval_score']) for h in history]
        return train_rewards, test_rewards


if __name__ == '__main__':
    agent = QAgent(10, 10, 10)