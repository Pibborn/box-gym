import time

import stable_baselines3
from gym import spaces
import gym
from stable_baselines3 import SAC
import torch.nn
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder
from agents.TrackingCallback import TrackingCallback
from stable_baselines3.common.evaluation import evaluate_policy

from DataExtraction.Extraction import init_Scale, init_Basketball, init_Orbit, init_FreeFall
from DataExtraction.Extraction import update_Scale_table, update_Basketball_table, update_Orbit_table, update_FreeFall_table
from DataExtraction.WandB import wandbCSVTracking
from agents.AgentInterface import Agent

SCALE = 0
BASKETBALL = 1
ORBIT = 2
FREEFALL = 3

class StableBaselinesAgent(Agent):
    def __init__(self, input_dim, output_dim, policy='MlpPolicy'):
        super().__init__(input_dim, output_dim)
        self.agent = None
        self.name = ''
        self.policy = policy
        pass

    def train_episode(self, env, verbose=1):  # todo
        pass

    def evaluate(self, env: gym.Env):
        #state = env.env_method("resetState")  # torch.tensor(env.reset())
        state = env.reset()
        R = 0  # return (sum of rewards)
        t = 0  # time step
        done = False
        #print(state)
        while not done:
            action, states = self.agent.predict(state, deterministic=True)
            """if env.actions == 2:   # todo: fix
                action = action[0]"""
            state, reward, done, _ = env.step(action)
            R += reward
            t += 1
            reset = t == 200
            if done or reset:
                break
        return R, state, action

    def create_model(self, train_env, verbose=1, use_sde=False):
        """Create the model with the given policy in the environment

        :type train_env: DummyVecEnv
        :param train_env: training environment
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
        action_size = self.output_dim  # train_env.action_space.low.size

        if type(train_env.observation_space) == gym.spaces.Dict:
            train_env.observation_space = self.convert_observation_space(train_env.observation_space, order=train_env.order)
        if type(test_env.observation_space) == gym.spaces.Dict:
            test_env.observation_space = self.convert_observation_space(test_env.observation_space, order=train_env.order)

        # train_env = DummyVecEnv([lambda: train_env])
        # test_env = DummyVecEnv([lambda: test_env])
        # !!!

        # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=config.reward_norm)
        # test_env = VecNormalize(test_env, norm_obs=True, norm_reward=config.reward_norm)
        # train_env = VecVideoRecorder(train_env, 'videos', record_video_trigger=lambda x: x % PRINT_EVERY == 0, video_length=200) # todo: Video

        wandb_callback = WandbCallback(gradient_save_freq=config.printevery,
                                       model_save_path="results/temp",
                                       verbose=0)
        train_success_callback = TrackingCallback(train_env, printfreq=PRINT_EVERY, num_eval_episodes=N_TRIALS,
                                                  is_test=False)
        test_success_callback = TrackingCallback(test_env, printfreq=PRINT_EVERY, num_eval_episodes=N_TRIALS,
                                                 is_test=True)

        self.agent = self.create_model(train_env, verbose=verbose, use_sde=sde)
        if not config.envname.lower() == "basketball":
            self.agent.learn(MAX_EPISODES, log_interval=PRINT_EVERY, eval_env=test_env, eval_freq=PRINT_EVERY,
                             # callback=[wandb_callback, train_success_callback, test_success_callback],  # todo: fix for Basketball
                             callback=wandb_callback,
                             eval_log_path='agents/temp')
        else:
            self.agent.learn(MAX_EPISODES, log_interval=PRINT_EVERY, eval_env=test_env, eval_freq=PRINT_EVERY,
                             callback=wandb_callback,
                             eval_log_path='agents/temp')

        # try to load the model & test it
        """self.agent.save("SAC_Model_test")   # location is just a placeholder for now, could be replaced with extra parameter
        del self.agent
        self.agent = SAC.load("SAC_Model_test")"""
        test_rewards, df = self.test_loop(test_env, config=config, verbose=verbose)
        # self.evaluate_model(test_env=test_env, config=config)
        return test_rewards, df

    def save_agent(self, location):
        self.agent.save_agent(location)
        return

    def load_agent(self, location):
        pass

    def evaluate_model(self, test_env, config):
        EPISODES = config.episodes
        """if not (type(test_env) == stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv):
            test_env = DummyVecEnv([lambda: test_env])  # convert the model"""
        # Evaluate the trained agent
        mean_reward, std_reward = evaluate_policy(self.agent, test_env, n_eval_episodes=EPISODES, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        return mean_reward

    def test_loop(self, test_env, config, verbose=1):  # todo: fix
        """Loop for the testing part --> only run tests on the test environment with the trained agent.

        :type test_env: gym.Env
        :param test_env: test environment
        :type config: Namespace
        :param config: configurations, set in the arguments
        :type verbose: int
        :param verbose: detailed information about testing?
        :rtype: list[float]
        :returns: train and test rewards for plotting"""
        MAX_EPISODES = config.episodes
        DISCOUNT_FACTOR = config.discount
        N_TRIALS = config.trials
        REWARD_THRESHOLD = config.threshold
        PRINT_EVERY = config.printevery

        if config.envname.lower() in {"scale", "scale_single", "scale_exp", "scale_draw"}:
            mode = 0
            box_number = config.placed + config.actions  # count number of boxes
            df = init_Scale(box_number=box_number, save_boxsize=True)
        elif config.envname.lower() == "basketball":
            mode = 1
            df = init_Basketball()  # todo
        elif config.envname.lower() == "orbit":
            mode = 2
            df = init_Orbit()  # todo
        elif config.envname.lower() == "freefall":
            mode = 3
            df = init_FreeFall()
        else:
            raise NotImplementedError()

        test_rewards = []
        test_matches = 0
        # action_size = self.output_dim  # train_env.action_space.low.size

        if type(test_env.observation_space) != gym.spaces.box.Box and not config.raw_pixels:  # if not already converted to Box
            test_env.observation_space = self.convert_observation_space(test_env.observation_space)

        # convert type to DummyVecEnv if not already done
        if mode != FREEFALL:
            if type(test_env) != DummyVecEnv and type(test_env) != VecNormalize:
                test_env = DummyVecEnv([lambda: test_env])
        # test_env = VecNormalize(test_env, norm_obs=True, norm_reward=config.reward_norm)
        """video_length = 480 # todo: turn on video recording
        test_env = VecVideoRecorder(test_env, "videos/",
                                    record_video_trigger=lambda x: x == 0, video_length=video_length,
                                    name_prefix=f"{self.name}-agent")

        test_env.reset()"""

        #todo: delete
        MAX_EPISODES = 200
        for episode in range(1, MAX_EPISODES + 1):
            test_reward, state, action = self.evaluate(env=test_env)
            #print(state)
            # state = test_env.env_method("resetState")
            if mode == FREEFALL:
                #max_time = test_env.get_attr("max_time")
                #start_height = test_env.get_attr("starting_distance")
                max_time = test_env.max_time
                starting_distance = test_env.starting_distance
                df = update_FreeFall_table(df=df, state=state, env=test_env, config=config, max_time=max_time, action=action, start_distance=starting_distance, index=test_matches)
                test_matches += 1
            else:
                state = state[0]
                start_position_x, start_position_y = state[0], state[1]
                if test_reward > 10:
                # if test_reward >= 1:
                    if mode == SCALE:
                        df = update_Scale_table(df=df, state=state, env=test_env, config=config, box_number=box_number,
                                                index=test_matches, action=action)
                    elif mode == BASKETBALL:
                        df = update_Basketball_table(df=df, state=state, env=test_env, config=config,
                                                     start_position_x=start_position_x, start_position_y=start_position_y,
                                                     action=action, index=test_matches)
                    else:
                        pass
                    test_matches += 1
            test_rewards.append(test_reward)
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            wandb.log({'test_rewards': mean_test_rewards})
            # stats = self.agent.get_statistics()
            # wandb.log({'average_q': stats[0][1]})
            # wandb.log({'loss': stats[1][1]})
            """if episode % PRINT_EVERY == 0:
                print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                print(self.agent.get_statistics())"""
            # todo: fix
            #if mean_test_rewards >= REWARD_THRESHOLD:
            #    print(f'Reached reward threshold in {episode} episodes')
            #    break
        print(
            f"Success rate of test episodes: {test_matches}/{MAX_EPISODES}={(test_matches / MAX_EPISODES * 100):,.2f}%")
        test_success_rate = test_matches / MAX_EPISODES
        wandb.log({'test_success_rate': test_success_rate})
        #test_env.close()
        return test_rewards, df
