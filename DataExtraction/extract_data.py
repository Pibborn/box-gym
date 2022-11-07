import argparse
import math

import gym.spaces
import numpy as np
import pandas as pd

from DataExtraction.Extraction import init_Scale, init_Basketball, init_Orbit
from DataExtraction.Extraction import update_Scale_table, update_Basketball_table, update_Orbit_table

from stable_baselines3 import SAC
import run_agent
from ScaleEnvironment.ScaleExperiment import rescale_movement
from agents.StableBaselinesAgents.SACAgent import SACAgent
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from agents.TrackingCallback import TrackingCallback
from stable_baselines3.common.evaluation import evaluate_policy
from DataExtraction.WandB import wandbCSVTracking
from ArgumentParser import create_argparser

SCALE = 0
BASKETBALL = 1
ORBIT = 2


def writeData(env, agent, config, mode=SCALE):
    if mode == SCALE:
        box_number = config.placed + config.actions  # count number of boxes
        df = init_Scale(box_number=box_number, save_boxsize=True)
    elif mode == BASKETBALL:
        df = init_Basketball()  # todo
    elif mode == ORBIT:
        df = init_Orbit()  # todo
    else:
        raise NotImplementedError()

    test_rewards = []
    test_matches = 0
    N_TRIALS = 100

    test_env = DummyVecEnv([lambda: env])
    # test_env = VecNormalize(test_env, norm_obs=True, norm_reward=config.reward_norm)

    # Start a W&B run to log data
    run = wandb.init(project="box-gym", entity=config.entity, config=config, sync_tensorboard=True)

    # wandb_callback = WandbCallback(gradient_save_freq=config.printevery, model_save_path="results/temp", verbose=0)

    for episode in range(1, config.episodes + 1):
        state = np.array(test_env.reset())  # torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0
        done = False
        while not done:
            action, states = agent.agent.predict(state, deterministic=True)
            state, reward, done, info = env.step(action[0])
            R += reward
            t += 1
            reset = t == 200
            if done or reset:
                break
        if R > 1:
            if mode == SCALE:
                df = update_Scale_table(df=df, state=state, config=config, box_number=box_number, index=test_matches)
            elif mode == BASKETBALL:
                df = update_Basketball_table(df=df)  # todo
            elif mode == ORBIT:
                df = update_Orbit_table(df=df)  # todo
            test_matches += 1

        test_rewards.append(R)
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        # run.log({'test_rewards': mean_test_rewards})

    print(
        f"Success rate of test episodes: {test_matches}/{config.episodes}={(test_matches / config.episodes * 100):,.2f}%")
    # print(df)
    df.to_csv(f"savedagents/extracted_data/{config.path}.csv")

    # Do the tracking of the CSV File
    run = wandbCSVTracking(run, f"savedagents/extracted_data/{config.path}.csv", config)

    # Finish the run (useful in notebooks)
    run.finish()
    return


# Scale functions
def writeScaleData(env, agent, config, box_number=2):
    # create DataFrame: first 1/3 columns are positions, the next 1/3 densities and the rest the sizes of the boxes
    df_dict = {}
    df_dict.update({f'Position {i + 1}': pd.Series(dtype='float') for i in range(box_number)})
    df_dict.update({f'Density {i + 1}': pd.Series(dtype='float') for i in range(box_number)})
    df_dict.update({f'Boxsize {i + 1}': pd.Series(dtype='float') for i in range(box_number)})
    df = pd.DataFrame(df_dict)

    test_rewards = []
    test_matches = 0
    N_TRIALS = 100
    # env.seed(1080)
    test_env = DummyVecEnv([lambda: env])
    # test_env = VecNormalize(test_env, norm_obs=True, norm_reward=config.reward_norm)

    # Start a W&B run to log data
    run = wandb.init(project="box-gym", entity=config.entity, config=config, sync_tensorboard=True)

    # wandb_callback = WandbCallback(gradient_save_freq=config.printevery, model_save_path="results/temp", verbose=0)

    for episode in range(1, config.episodes + 1):
        state = np.array(test_env.reset())  # torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0
        done = False
        while not done:
            action, states = agent.agent.predict(state, deterministic=True)
            state, reward, done, info = env.step(action[0])
            R += reward
            t += 1
            reset = t == 200
            if done or reset:
                break
        if R > 1:
            density_index = 2 + box_number
            size_index = 2 + 2 * box_number
            if config.normalize:
                # only can access positions of placed boxes here
                positions = rescale_movement([-1, 1], state[:density_index - 2],
                                             [-20, 20])  # todo: fix normalized data extraction
                densities = rescale_movement([0, 1], state[density_index:size_index], [4, 6])
                sizes = rescale_movement([0, 1], state[size_index:], [0.8, 1.2])
            else:
                positions = state[:density_index - 2]
                densities = state[density_index:size_index]
                sizes = state[size_index:]

            df.loc[test_matches] = np.concatenate((positions, densities, sizes))
            test_matches += 1
            # print(state)
            # df2 = pd.DataFrame([[]], columns=list('AB'))
        test_rewards.append(R)
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        # run.log({'test_rewards': mean_test_rewards})

    print(
        f"Success rate of test episodes: {test_matches}/{config.episodes}={(test_matches / config.episodes * 100):,.2f}%")
    # print(df)
    df.to_csv(f"savedagents/extracted_data/{config.path}.csv")

    # Do the tracking of the CSV File
    run = wandbCSVTracking(run, f"savedagents/extracted_data/{config.path}.csv", config)

    # Finish the run (useful in notebooks)
    run.finish()
    return


def readScaleData(env, config):
    df = pd.read_csv(f"savedagents/extracted_data/{config.path}.csv")
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index()  # make sure indexes pair with number of rows
    env.reset()
    for index, row in df.iterrows():
        env.reset()
        pos1, pos2 = row['Position 1'], row['Position 2']
        den1, den2 = row['Density 1'], row['Density 2']
        size1, size2 = row['Boxsize 1'], row['Boxsize 2']
        env.deleteBox(env.boxA)
        env.deleteBox(env.boxB)
        env.boxA = env.createBox(pos_x=pos1, density=den1, boxsize=size1)
        env.boxB = env.createBox(pos_x=pos2, density=den2, boxsize=size2)
        env.step(None)
    return


# Basketball functions
def writeBasketballData(env, agent, config):
    # create DataFrame for Basketball
    df = pd.DataFrame({'x-Position Ball': pd.Series(dtype='float'),
                       'y-Position Ball': pd.Series(dtype='float'),
                       'Force vector x': pd.Series(dtype='float'),
                       'Force vector y': pd.Series(dtype='float'),
                       'Radius': pd.Series(dtype='float'),
                       'Density': pd.Series(dtype='float'),
                       'x-Position Basket': pd.Series(dtype='float'),
                       'y-Position Basket': pd.Series(dtype='float'),
                       'Radius Basket': pd.Series(dtype='float'),
                       })
    """df = pd.DataFrame({'x-Position Ball': pd.Series(dtype='float'),
                       'y-Position Ball': pd.Series(dtype='float'),
                       'Angle': pd.Series(dtype='float'),
                       'Velocity': pd.Series(dtype='float'),
                       'Radius': pd.Series(dtype='float'),
                       'Density': pd.Series(dtype='float'),
                       'x-Position Basket': pd.Series(dtype='float'),
                       'y-Position Basket': pd.Series(dtype='float'),
                       'Radius Basket': pd.Series(dtype='float'),
                       })"""

    test_matches = 0
    # env.seed(1080)
    test_env = DummyVecEnv([lambda: env])
    # test_env = VecNormalize(test_env, norm_obs=True, norm_reward=config.reward_norm)

    for episode in range(1, config.episodes + 1):
        state = test_env.reset()  # torch.tensor(env.reset())
        start_position_x, start_position_y = state[0][0], state[0][1]
        print(start_position_x, start_position_y)

        R = 0  # return (sum of rewards)
        t = 0
        done = False
        while not done:
            action, states = agent.agent.predict(state, deterministic=True)
            state, reward, done, info = env.step(action[0])
            R += reward
            t += 1
            reset = t == env.max_time_steps
            if done or reset:
                break
        if R >= 100:
            if config.normalize:
                """state = rescale_movement([np.array([0, 0, -1, -1, 0, 0, 0, 0, 0]), np.array([1 for _ in range(9)])],
                                         state,
                                         [np.array([0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5]),
                                          np.array([env.world_width, env.world_height, np.pi, 10, 1.5, 6,
                                                    env.world_width, env.world_height, 3])])"""
                # rescaled_action = rescale_movement([0, 1], action, [0, 15])
                state = rescale_movement([np.array([0, 0, -1, -1, 0, 0, 0, 0, 0]), np.array([1 for _ in range(9)])],
                                         np.concatenate(
                                             (np.array([start_position_x, start_position_y]), action[0], state[4:])),
                                         [np.array([0, 0, 0, 0, 0.5, 4, 0, 0, 0.5]),
                                          np.array([env.world_width, env.world_height, 15, 15, 1.5, 6,
                                                    env.world_width, env.world_height, 3])])

            df.loc[test_matches] = np.array(state)
            test_matches += 1
            # print(state)
            # df2 = pd.DataFrame([[]], columns=list('AB'))

    print(
        f"Success rate of test episodes: {test_matches}/{config.episodes}={(test_matches / config.episodes * 100):,.2f}%")
    # print(df)
    df.to_csv(f"savedagents/extracted_data/{config.path}.csv")
    return


def readBasketballData(env, config):
    # todo
    return


# Main
if __name__ == '__main__':
    args = create_argparser()

    _, test_env = run_agent.create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                        random_densities=args.random_densities, random_boxsizes=args.random_boxsizes,
                                        normalize=args.normalize, placed=args.placed, actions=args.actions,
                                        sides=args.sides, raw_pixels=args.raw_pixels,
                                        random_ball_position=args.random_ball_position, walls=args.walls,
                                        random_density=args.random_density or args.random_densities,
                                        random_basket=args.random_basket, random_ball_size=args.random_ball_size)
    input_dim, output_dim = run_agent.get_env_dims(test_env)
    if args.agent == 'sac':
        agent = SACAgent(input_dim, output_dim)
        if type(test_env.observation_space) != gym.spaces.Box:
            test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
        if args.location == "":
            args.location = 'SAC_Model'
        args.location = f"savedagents/models/{args.location}"
        agent.agent = SAC.load(args.location, env=test_env)
    else:

        raise NotImplementedError("not implemented")

    if args.envname == 'Basketball':
        if args.mode == 1:
            writeBasketballData(env=test_env, agent=agent, config=args)

        else:
            readBasketballData(env=test_env, config=args)
    else:
        if args.mode == 1:
            # writeScaleData(env=test_env, agent=agent, config=args, box_number=args.placed + args.actions)
            writeData(env=test_env, agent=agent, config=args, mode=SCALE)
        else:
            readScaleData(env=test_env, config=args, box_number=args.placed + args.actions)
