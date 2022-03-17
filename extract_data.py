import argparse
import math

import pandas as pd

from stable_baselines3 import SAC
import run_agent
from ScaleEnvironment.ScaleExperiment import rescale_movement
from agents.StableBaselinesAgents.SACAgent import SACAgent
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def writeData(env, agent, config):
    df = pd.DataFrame({'Position 1': pd.Series(dtype='float'),
                       'Position 2': pd.Series(dtype='float'),
                       'Density 1': pd.Series(dtype='float'),
                       'Density 2': pd.Series(dtype='float'),
                       'Boxsize 1': pd.Series(dtype='float'),
                       'Boxsize 2': pd.Series(dtype='float')})
    test_matches = 0
    env.seed(1080)
    test_env = DummyVecEnv([lambda: env])
    #test_env = VecNormalize(test_env, norm_obs=True, norm_reward=config.reward_norm)

    for episode in range(1, config.episodes + 1):
        state = test_env.reset()  # torch.tensor(env.reset())
        R = 0  # return (sum of rewards)
        t = 0
        done = False
        while not done:
            action, states = agent.agent.predict(state, deterministic=True)
            #pos1 = test_env.venv.boxA.position[0] / math.cos(test_env.venv.bar.angle)
            #state = test_env.get_original_obs()
            if config.normalize:
                pos1 = rescale_movement([-1, 1], state[0][0], [-20, 20])
                den1 = rescale_movement([0, 1], state[0][4], [4, 6])
                den2 = rescale_movement([0, 1], state[0][5], [4, 6])
                size1 = rescale_movement([0, 1], state[0][6], [0.8, 1.2])
                size2 = rescale_movement([0, 1], state[0][7], [0.8, 1.2])
            else:
                pos1 = state[0][0]
                den1 = state[0][4]
                den2 = state[0][5]
                size1 = state[0][6]
                size2 = state[0][7]
            state, reward, done, info = env.step(action[0])
            R += reward
            t += 1
            reset = t == 200
            if done or reset:
                break
        if R > 1:
            if env.actions == 2:
                df.loc[test_matches] = [action[0][0], action[0][1], den1, den2, size1, size2]
            else:  # different things to track
                df.loc[test_matches] = [pos1, action[0][0], den1, den2, size1, size2]
            test_matches += 1
            #print(state)
            #df2 = pd.DataFrame([[]], columns=list('AB'))

    print(
        f"Success rate of test episodes: {test_matches}/{config.episodes}={(test_matches / config.episodes * 100):,.2f}%")
    #print(df)
    df.to_csv(f"savedagents/extracted_data/{config.path}.csv")
    return

def readData(env, config):
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

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str, default='scale_single')
    parser.add_argument('--agent', type=str, default='sac')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random_densities', action='store_true')
    parser.add_argument('--random_boxsizes', action='store_true')
    parser.add_argument('--rendering', action='store_true')
    parser.add_argument('--location', type=str, default="")
    parser.add_argument('--reward-norm', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--path', type=str, default='results')
    parser.add_argument('--mode', type=int, default=1)
    args = parser.parse_args()

    _, test_env = run_agent.create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                        random_densities=args.random_densities, random_boxsizes=args.random_boxsizes,
                                        normalize=args.normalize)
    input_dim, output_dim = run_agent.get_env_dims(test_env)
    if args.agent == 'sac':
        agent = SACAgent(input_dim, output_dim)
        test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
        if args.location == "":
            args.location = 'SAC_Model'
        args.location = f"savedagents/models/{args.location}"
        agent.agent = SAC.load(args.location, env=test_env)
    else:

        raise NotImplementedError("not implemented")

    if args.mode == 1:
        writeData(env=test_env, agent=agent, config=args)

    else:
        readData(env=test_env, config=args)