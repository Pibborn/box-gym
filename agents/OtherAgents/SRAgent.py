import numpy as np
import torch
import wandb

from ScaleEnvironment.ScaleExperiment import rescale_movement
from agents.AgentInterface import Agent
from math import sin, cos


def inv(x):
    if x == 0:
        raise ZeroDivisionError
    return 1 / x


class SRAgent(Agent):
    def __init__(self, input_dim, output_dim, function):
        super().__init__(input_dim, output_dim)
        self.function = function

    def train_episode(self, env, verbose=1):
        pass

    def evaluate(self, env):
        state = env.reset()
        R = 0  # return (sum of rewards)
        t = 0  # time step
        done = False
        while not done:
            x1 = rescale_movement([-1., 1.], state[0], [-20., 20.])
            x2 = rescale_movement([0., 1.], state[4], [0., 6.])  # other possibility: [4.,6.] -> [-1,1]
            x3 = rescale_movement([0., 1.], state[5], [0., 6.])
            x4 = 1.0  # box size of left box  #todo: include box sizes
            x5 = 1.0  # box size of right box
            action = eval(self.function)
            # print(action)
            state, reward, done, _ = env.step(action)
            R += reward
            t += 1
            reset = t == 200
            #self.agent.observe(state, reward, done, reset)
            if done or reset:
                break
        return R

    def train_loop(self, train_env, test_env, config, verbose=1, sde=False, only_testing=False):
        pass

    def test_loop(self, test_env, config, verbose=1):
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery
        test_rewards = []
        test_matches = 0

        for episode in range(1, MAX_EPISODES + 1):
            test_reward = self.evaluate(env=test_env)
            if test_reward >= 1:
                test_matches += 1
            test_rewards.append(test_reward)
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            wandb.log({'test_rewards': mean_test_rewards})
            if mean_test_rewards >= REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
        print(
            f"Success rate of test episodes: {test_matches}/{MAX_EPISODES}={(test_matches / MAX_EPISODES * 100):,.2f}%")
        test_success_rate = test_matches / MAX_EPISODES
        wandb.log({'test_success_rate': test_success_rate})
        return test_rewards

    def save_agent(self, location):
        pass

    def load_agent(self, location):
        pass
