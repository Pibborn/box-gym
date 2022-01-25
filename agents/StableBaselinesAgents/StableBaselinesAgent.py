import stable_baselines3
from gym import spaces
import gym
from stable_baselines3 import SAC
import torch.nn
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder

from agents.AgentInterface import Agent


class StableBaselinesAgent(Agent):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.agent = None
        pass

    def train_episode(self, env, verbose=1):  # todo
        pass

    def evaluate(self, env: gym.Env):
        state = torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0  # time step
        done = False
        while not done:
            action, states = self.agent.predict(state)
            state, reward, done, _ = env.step(action)
            R += reward
            t += 1
            reset = t == 200
            if done or reset:
                break
        return R

    def create_model(self, train_env, policy='MlpPolicy', verbose=1, use_sde=False):
        """Create the model with the given policy in the environment

        :type train_env: DummyVecEnv
        :param train_env: training environment
        :type policy: str
        :param policy: policy
        :type verbose: int
        :param verbose: print out every information
        :type use_sde: bool
        :param use_sde: use sde (standard dependent exploration)?
        :rtype:
        :returns: model (i.e. SAC or A2C model)"""
        return None

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

        train_env.observation_space = self.convert_observation_space(train_env.observation_space)

        train_env = DummyVecEnv([lambda: train_env])
        #train_env = VecVideoRecorder(train_env, 'videos', record_video_trigger=lambda x: x % 1000 == 0, video_length=200) # todo: Video

        wandb_callback = WandbCallback(gradient_save_freq=100,
                                       model_save_path="results/temp",
                                       verbose=0)

        eval_callback = EvalCallback(train_env, best_model_save_path='results/temp',
                                     log_path='./logs/', eval_freq=100,
                                     deterministic=True, render=False)

        self.agent = self.create_model(train_env, policy='MlpPolicy', verbose=verbose, use_sde=sde)
        self.agent.learn(MAX_EPISODES, callback=eval_callback)  # todo: extract train rewards and mean train reward and save them on wandb

        for episode in range(1, MAX_EPISODES + 1):
            test_reward = self.evaluate(test_env)
            if test_reward >= 1:
                test_matches += 1
            test_rewards.append(test_reward)
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            wandb.log({'test_rewards': mean_test_rewards})
            #stats = self.agent.get_statistics()
            #wandb.log({'average_q': stats[0][1]})
            #wandb.log({'loss': stats[1][1]})
            """if episode % PRINT_EVERY == 0:
                if not only_testing:
                    print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                elif only_testing:
                    print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                print(self.agent.get_statistics())"""
            if mean_test_rewards >= REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
        """train_success_rate = train_matches / MAX_EPISODES
        wandb.log({'train_success_rate': train_success_rate})
        print(f"\nSuccess rate of train episodes: {train_matches}/{MAX_EPISODES}={(train_matches / MAX_EPISODES) * 100:,.2f}%")"""
        print(f"Success rate of test episodes: {test_matches}/{MAX_EPISODES}={(test_matches / MAX_EPISODES * 100):,.2f}%")
        test_success_rate = test_matches / MAX_EPISODES
        wandb.log({'test_success_rate': test_success_rate})
        return train_rewards, test_rewards

    def save_agent(self, location):
        self.agent.save_agent(location)
        return

    def load_agent(self, location):
        pass