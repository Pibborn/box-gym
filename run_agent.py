import time

import gym
import matplotlib

from agents.OtherAgents.SRAgent import SRAgent
from agents.StableBaselinesAgents.A2CAgent import A2CAgent
# from agents.StableBaselinesAgents.CustomAgent import CustomAgent
from agents.StableBaselinesAgents.PPOAgent import PPOAgent
from agents.StableBaselinesAgents.SACAgent import SACAgent
from agents.StableBaselinesAgents.HERAgent import HERAgent
from environments.BasketballEnvironment import BasketballEnvironment
# from environments.Pendulum import PendulumEnv, RGBArrayAsObservationWrapper
from environments.OrbitEnvironment import OrbitEnvironment
from environments.FreeFallEnvironment import FreeFallEnvironment
from DataExtraction.WandB import wandbCSVTracking
from ArgumentParser import create_argparser

matplotlib.rcParams['backend'] = 'WebAgg'
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

matplotlib.rcParams['backend'] = 'WebAgg'
import matplotlib.pyplot as plt

from environments.GymEnv import GymEnv
from ScaleEnvironment.Scale import Scale
from ScaleEnvironment.ScaleExperiment import ScaleExperiment
from ScaleEnvironment.ScaleDraw import ScaleDraw
from stable_baselines3.sac import SAC
from stable_baselines3.a2c import A2C
from gym.spaces import Dict
import argparse
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from rich.traceback import install

install(show_locals=True)


def create_envs(env_str='', seed=42, do_render=True, random_densities=False, random_boxsizes=False, normalize=False,
                placed=1, actions=1, sides=2, raw_pixels=False, use_own_render_function=False, walls=0,
                random_density=False, random_ball_size=False, random_basket=False, random_ball_position=False,
                random_planet_position=False, random_gravity=False,
                random_satellite_position=False, random_satellite_density=False, random_satellite_size=False,
                random_ball_height=True, use_seconds=False):
    allowed_strings = ['scale', 'scale_exp', 'scale_single', 'scale_draw', 'basketball', 'orbit', 'freefall']
    if str(env_str).lower() not in allowed_strings:
        raise AssertionError(f"Environment name {env_str} not in allowed list {allowed_strings}")
    env_str = env_str.lower()
    if env_str == 'scale':
        train_env = Scale(rendering=do_render, random_densities=random_densities,
                          random_boxsizes=random_boxsizes, normalize=normalize,
                          placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels)
        test_env = Scale(rendering=do_render, random_densities=random_densities,
                         random_boxsizes=random_boxsizes, normalize=normalize,
                         placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels)
    elif env_str == 'scale_exp':
        train_env = ScaleExperiment(rendering=do_render, random_densities=random_densities,
                                    random_boxsizes=random_boxsizes, actions=2, normalize=normalize, boxes=2)
        test_env = ScaleExperiment(rendering=do_render, random_densities=random_densities,
                                   random_boxsizes=random_boxsizes, actions=2, normalize=normalize, boxes=2)
    elif env_str == 'scale_single':
        train_env = ScaleExperiment(rendering=do_render, random_densities=random_densities,
                                    random_boxsizes=random_boxsizes, actions=1, normalize=normalize, boxes=2)
        test_env = ScaleExperiment(rendering=do_render, random_densities=random_densities,
                                   random_boxsizes=random_boxsizes, actions=1, normalize=normalize, boxes=2)
    elif env_str == 'scale_draw':
        train_env = ScaleDraw(rendering=do_render, random_densities=random_densities,
                              random_boxsizes=random_boxsizes, normalize=normalize,
                              placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels,
                              use_own_render_function=use_own_render_function)
        """test_env = ScaleDraw(rendering=do_render, random_densities=random_densities,
                             random_boxsizes=random_boxsizes, normalize=normalize,
                             placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels,
                             use_own_render_function=use_own_render_function)"""
        test_env = ScaleDraw(rendering=True, random_densities=random_densities,
                             random_boxsizes=random_boxsizes, normalize=normalize,
                             placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels,
                             use_own_render_function=use_own_render_function)
    elif env_str == 'basketball':
        train_env = BasketballEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels, walls=walls,
                                          random_density=random_density or random_densities,
                                          random_ball_size=random_ball_size or random_boxsizes,
                                          random_basket=random_basket, random_ball_position=random_ball_position,
                                          random_gravity=random_gravity)
        #test_env = BasketballEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels, walls=walls,
        test_env = BasketballEnvironment(rendering=True, normalize=normalize, raw_pixels=raw_pixels, walls=walls,
                                         random_density=random_density or random_densities,
                                         random_ball_size=random_ball_size or random_boxsizes,
                                         random_basket=random_basket, random_ball_position=random_ball_position,
                                         random_gravity=random_gravity)
    elif env_str == 'orbit':
        train_env = OrbitEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels,
                                     random_planet_position=random_planet_position, random_gravity=random_gravity,
                                     random_satellite_position=random_satellite_position,
                                     random_satellite_size=random_satellite_size,
                                     random_satellite_density=random_satellite_density)
        test_env = OrbitEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels,
                                    random_planet_position=random_planet_position, random_gravity=random_gravity,
                                    random_satellite_position=random_satellite_position,
                                    random_satellite_size=random_satellite_size,
                                    random_satellite_density=random_satellite_density)
    elif env_str == 'freefall':
        train_env = FreeFallEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels,
                                        random_ball_size=random_ball_size, random_density=random_density,
                                        random_gravity=random_gravity, random_ball_height=random_ball_height,
                                        use_seconds=use_seconds)
        test_env = FreeFallEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels,
                                        random_ball_size=random_ball_size, random_density=random_density,
                                        random_gravity=random_gravity, random_ball_height=random_ball_height,
                                        use_seconds=use_seconds)
    else:
        train_env = GymEnv(env_str)
        train_env = train_env.create()
        test_env = GymEnv(env_str)
        test_env = test_env.create()
    if seed:
        train_env.seed(seed)
        test_env.seed(seed + 1)
        set_random_seed(seed)
    return train_env, test_env


def get_env_dims(env):
    if type(env.action_space) is not Dict:
        out_dim = int(env.action_space.shape[0])
    else:
        out_dim = len(env.action_space)
    if type(env.observation_space) is not Dict:
        in_dim = env.observation_space.shape
    else:
        in_dim = len(env.observation_space)
    return in_dim, out_dim

def create_agent(agentname, location, args, test_env=None):
    if not args.test:
        if agentname.lower() == 'sac':
            agent = SACAgent(input_dim, output_dim, lr=args.lr,
                             policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if location == "":
                location = "SAC_Model"
        elif agentname.lower() == 'a2c':
            agent = A2CAgent(input_dim, output_dim, lr=args.lr,
                             policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if location == "":
                location = "A2C_Model"
        elif agentname.lower() == 'ppo':
            agent = PPOAgent(input_dim, output_dim, lr=args.lr,
                             policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if location == "":
                location = "A2C_Model"
        elif agentname.lower() == 'her':  # todo: delete
            agent = HERAgent(input_dim, output_dim,
                             lr=args.lr, )  # policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if location == "":
                location = "HER_Model"
        elif agentname.lower() == 'custom':
            """ agent = CustomAgent(input_dim, output_dim, lr=args.lr,) # policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
                if args.location == "":
                    args.location = "Custom_Model" """
            pass
        else:
            raise ValueError('Agent string {} not recognized'.format(agentname))
        args.location = f"savedagents/models/{args.location}"
        # agent = QAgent(input_dim, output_dim, gamma=args.discount, lr=args.lr)
        # agent = VanillaGradMLP(input_dim, 100, output_dim, dropout=args.dropout, uses_scale=args.envname=='scale',
        #                     scale_exp=args.envname=='scale_exp')
        return agent, location
    else:
        if agentname == 'sac':
            agent = SACAgent(input_dim, output_dim, lr=args.lr)
            if type(test_env.observation_space) != gym.spaces.Box:
                test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
            location = f"savedagents/models/{location if location != '' else 'SAC_Model'}"
            agent.agent = SAC.load(location if location != '' else 'SAC_Model', env=test_env)
            # agent.agent.load_replay_buffer(f"{args.location}_replay_buffer")
        elif agentname == 'a2c':
            agent = A2CAgent(input_dim, output_dim, lr=args.lr)
            test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
            location = f"savedagents/models/{args.location if location != '' else 'A2C_Model'}"
            agent.agent = A2C.load(args.location if location != '' else 'A2C_Model', env=test_env)
            # agent.agent.load_replay_buffer(f"{args.location}_replay_buffer")
        elif args.agent == 'custom':
            """agent = CustomAgent(input_dim, output_dim, lr=args.lr)
            test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
            location = f"savedagents/models/{args.location if args != '' else 'Custom_Model'}"
            agent.agent = SAC.load(args.location if args != '' else 'Custom_Model', env=test_env)
            # agent.agent.load_replay_buffer(f"{args.location}_replay_buffer")"""
            pass
        elif args.agent == 'sr':
            formula = "(x1 * (x2 * -0.99871546) * inv(x3))"
            formula = "((x1 * -0.998) / (x3 / x2))"  # "div(mul(X0, -0.998), div(X2, X1))"
            formula = "((-0.005 - x1) / (x3 / x2))"
            agent = SRAgent(input_dim, output_dim, function=formula)
        else:
            raise ValueError('Agent string {} not recognized'.format(args.agent))
        # agent.load_agent(args.location)
        return agent, location


def plot_rewards(train_rewards, test_rewards, threshold):
    plt.figure(figsize=(12, 8))
    try:  # if QAgent is used
        xs, ys = zip(*test_rewards)
        xs = (0,) + xs
        ys = (0,) + ys
        plt.plot(xs, ys, label='Test Reward')
        plt.hlines(threshold, 0, xs[-1], color='r')
    except Exception as e:  # use this for the VanillaGradMLP agent
        plt.plot(test_rewards, label='Test Reward')
        plt.hlines(threshold, 0, len(test_rewards), color='r')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_test_rewards(test_rewards, threshold):
    plt.figure(figsize=(12, 8))
    # plt.title(f"Randomness = {randomness}, discount = {discount}, dropout = {dropout}", fontdict={'fontsize': 20})
    try:  # if QAgent is used
        xs, ys = zip(*test_rewards)
        xs = (0,) + xs
        ys = (0,) + ys
        plt.plot(xs, ys, label='Test Reward')
        plt.hlines(threshold, 0, xs[-1], color='r')
    except Exception as e:  # use this for the VanillaGradMLP agent
        plt.plot(test_rewards, label='Test Reward')
        plt.hlines(threshold, 0, len(test_rewards), color='r')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    args = create_argparser()

    if not args.disable_xvfb:
        from xvfbwrapper import Xvfb

        vdisplay = Xvfb()
        vdisplay.start()

    wandb.tensorboard.patch(root_logdir="<logging_directory>")
    run = wandb.init(project="box-gym", entity=args.entity, config=args, sync_tensorboard=True)

    if not args.test:  # train + test new agent
        train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                          random_densities=args.random_densities, random_boxsizes=args.random_boxsizes,
                                          normalize=args.normalize, placed=args.placed, actions=args.actions,
                                          sides=args.sides, raw_pixels=args.raw_pixels,
                                          use_own_render_function=args.use_own_render_function, walls=args.walls,
                                          random_density=args.random_density, random_ball_size=args.random_ball_size,
                                          random_basket=args.random_basket,
                                          random_ball_position=args.random_ball_position,
                                          random_satellite_position=args.random_satellite_position,
                                          random_planet_position=args.random_planet_position,
                                          random_satellite_size=args.random_satellite_size,
                                          random_satellite_density=args.random_satellite_size,
                                          random_gravity=args.random_gravity,
                                          random_ball_height=args.random_ball_height, use_seconds=args.use_seconds)
        input_dim, output_dim = get_env_dims(train_env)

        # agent, args.location = create_agent(agentname=args.agent, location=args.location, args=args)
        # todo: fix bug that it's saved wrong
        agent, args.location = create_agent(agentname=args.agent, location=f"savedagents/models/{args.location}", args=args)

        test_rewards, df = agent.train_loop(train_env, test_env, args, verbose=1, only_testing=False)
        # save the trained agent
        if args.overwriting:
            agent.agent.save(args.location)
            # agent.save_agent(args.location)   # could also do this...
            print(f"Saved model {args.agent} to location {args.location}.zip")
            # agent.agent.save_replay_buffer(f"{args.location}_replay_buffer")
            # print(f"Saved replay buffer from training to location {args.location}_replay_buffer.pkl")

        # save the csv file
        df.to_csv(f"savedagents/extracted_data/{args.path}.csv")

        # Do the tracking of the CSV File
        run = wandbCSVTracking(run, f"savedagents/extracted_data/{args.path}.csv", args)

    else:  # load old agent and test him
        # load the agent
        train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                          random_densities=args.random_densities, random_boxsizes=args.random_boxsizes,
                                          normalize=args.normalize, placed=args.placed, actions=args.actions,
                                          sides=args.sides, raw_pixels=args.raw_pixels,
                                          use_own_render_function=args.use_own_render_function, walls=args.walls,
                                          random_density=args.random_density, random_ball_size=args.random_ball_size,
                                          random_basket=args.random_basket,
                                          random_ball_position=args.random_ball_position,
                                          random_satellite_position=args.random_satellite_position,
                                          random_planet_position=args.random_planet_position,
                                          random_satellite_size=args.random_satellite_size,
                                          random_satellite_density=args.random_satellite_size,
                                          random_gravity=args.random_gravity,
                                          random_ball_height=args.random_ball_height, use_seconds=args.use_seconds)
        input_dim, output_dim = get_env_dims(test_env)

        agent, args.location = create_agent(agentname=args.agent, location=args.location, args=args, test_env=test_env)
        # test_env = agent.agent.get_env()  # maybe the better solution?

        print(f"Loaded agent from Model {args.agent} from location {args.location}")

        # use the loaded agent
        mean_test_rewards, df = agent.test_loop(test_env=test_env, config=args, verbose=1)
        # mean_test_rewards = agent.evaluate_model(test_env=test_env, config=args)

        df.to_csv(f"savedagents/extracted_data/{args.path}.csv")

        # Do the tracking of the CSV File
        run = wandbCSVTracking(run, f"savedagents/extracted_data/{args.path}.csv", args)

    end_time = time.time()
    print(end_time - start_time)

    # Finish the run (useful in notebooks)
    run.finish()
