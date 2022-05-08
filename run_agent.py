import time

import matplotlib

from agents.OtherAgents.SRAgent import SRAgent
from agents.StableBaselinesAgents.A2CAgent import A2CAgent
# from agents.StableBaselinesAgents.CustomAgent import CustomAgent
from agents.StableBaselinesAgents.PPOAgent import PPOAgent
from agents.StableBaselinesAgents.SACAgent import SACAgent
from agents.StableBaselinesAgents.HERAgent import HERAgent
from environments.BasketballEnvironment import BasketballEnvironment
from environments.Pendulum import PendulumEnv, RGBArrayAsObservationWrapper

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
                placed=1, actions=1, sides=2, raw_pixels=False, walls=0,
                random_density=False, random_ball_size=False, random_basket=False, random_ball_position=False):
    allowed_strings = ['scale', 'scale_exp', 'scale_single', 'scale_draw', '']
    # if str(env_str) not in allowed_strings:
    #    raise AssertionError(f"Environment name {env_str} not in allowed list {allowed_strings}")
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
                              placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels)
        test_env = ScaleDraw(rendering=do_render, random_densities=random_densities,
                             random_boxsizes=random_boxsizes, normalize=normalize,
                             placed=placed, actions=actions, sides=sides, raw_pixels=raw_pixels)
    elif env_str == 'basketball':
        train_env = BasketballEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels, walls=walls,
                                          random_density=random_density or random_densities,
                                          random_ball_size=random_ball_size or random_boxsizes,
                                          random_basket=random_basket, random_ball_position=random_ball_position)
        test_env = BasketballEnvironment(rendering=do_render, normalize=normalize, raw_pixels=raw_pixels, walls=walls,
                                         random_density=random_density or random_densities,
                                         random_ball_size=random_ball_size or random_boxsizes,
                                         random_basket=random_basket, random_ball_position=random_ball_position)
    elif env_str == 'pendulum':
        train_env = PendulumEnv(g=9.81, rendering=do_render)
        test_env = PendulumEnv(g=9.81, rendering=do_render)
        if args.raw_pixels:
            train_env = RGBArrayAsObservationWrapper(train_env)
            test_env = RGBArrayAsObservationWrapper(test_env)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str, default='scale_single')
    parser.add_argument('--agent', type=str, default='sac')
    parser.add_argument('--seed', type=int, default=42)  # old default: 42
    parser.add_argument('--episodes', type=int, default=10000)  # old default: 1000
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--printevery', type=int, default=500)  # old default: 10
    parser.add_argument('--discount', type=float, default=0.99)  # old default: 0.99
    parser.add_argument('--threshold', type=float, default=20.1)  # old default: 475
    parser.add_argument('--dropout', type=float, default=0.2)  # old default: 0.2
    parser.add_argument('--random_densities', action='store_true')
    parser.add_argument('--random_boxsizes', action='store_true')
    parser.add_argument('--rendering', action='store_true')
    parser.add_argument('--overwriting', action='store_true')
    parser.add_argument('--entity', type=str, default='jgu-wandb')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--reward-norm', action='store_true')
    parser.add_argument('--disable_xvfb', action='store_true')
    parser.add_argument('--location', type=str, default="")
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--placed', type=int, default=1)
    parser.add_argument('--actions', type=int, default=1)
    parser.add_argument('--sides', type=int, default=2)
    parser.add_argument('--raw_pixels', action='store_true')
    # additional basketball settings
    parser.add_argument('--random_density', action='store_true')  # equivalent to --random_densities flag
    parser.add_argument('--random_ball_size', action='store_true')  # equivalent to --random_boxsize flag
    parser.add_argument('--random_basket', action='store_true')
    parser.add_argument('--random_ball_position', action='store_true')
    parser.add_argument('--walls', type=int, default=0)
    args = parser.parse_args()

    if not args.disable_xvfb:
        from xvfbwrapper import Xvfb

        vdisplay = Xvfb()
        vdisplay.start()

    wandb.init(project="box-gym", entity=args.entity, config=args, sync_tensorboard=True)

    if not args.test:  # train + test new agent
        train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                          random_densities=args.random_densities, random_boxsizes=args.random_boxsizes,
                                          normalize=args.normalize, placed=args.placed, actions=args.actions,
                                          sides=args.sides, raw_pixels=args.raw_pixels, walls=args.walls,
                                          random_density=args.random_density, random_ball_size=args.random_ball_size,
                                          random_basket=args.random_basket, random_ball_position=args.random_ball_position)
        input_dim, output_dim = get_env_dims(train_env)
        # agent = QAgent(input_dim, output_dim, gamma=args.discount, lr=args.lr)
        if args.agent.lower() == 'sac':
            agent = SACAgent(input_dim, output_dim, lr=args.lr,
                             policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if args.location == "":
                args.location = "SAC_Model"
        elif args.agent.lower() == 'a2c':
            agent = A2CAgent(input_dim, output_dim, lr=args.lr,
                             policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if args.location == "":
                args.location = "A2C_Model"
        elif args.agent.lower() == 'ppo':
            agent = PPOAgent(input_dim, output_dim, lr=args.lr,
                             policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if args.location == "":
                args.location = "A2C_Model"
        elif args.agent.lower() == 'her':  # todo: delete
            agent = HERAgent(input_dim, output_dim,
                             lr=args.lr, )  # policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
            if args.location == "":
                args.location = "HER_Model"
        elif args.agent.lower() == 'custom':
            pass
            """
                agent = CustomAgent(input_dim, output_dim, lr=args.lr,) # policy='MlpPolicy' if not args.raw_pixels else 'CnnPolicy')
                if args.location == "":
                    args.location = "Custom_Model"
            """
        else:
            raise ValueError('Agent string {} not recognized'.format(args.agent))
        args.location = f"savedagents/models/{args.location}"
        # agent = VanillaGradMLP(input_dim, 100, output_dim, dropout=args.dropout, uses_scale=args.envname=='scale',
        #                     scale_exp=args.envname=='scale_exp')
        agent.train_loop(train_env, test_env, args, verbose=1, only_testing=False)
        # save the trained agent
        if args.overwriting:
            agent.agent.save(args.location)
            # agent.save_agent(args.location)   # could also do this...
            print(f"Saved model {args.agent} to location {args.location}.zip")
            # agent.agent.save_replay_buffer(f"{args.location}_replay_buffer")
            # print(f"Saved replay buffer from training to location {args.location}_replay_buffer.pkl")

    else:  # load old agent and test him
        # load the agent
        train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                          random_densities=args.random_densities, random_boxsizes=args.random_boxsizes,
                                          normalize=args.normalize, placed=args.placed, actions=args.actions,
                                          sides=args.sides)
        input_dim, output_dim = get_env_dims(test_env)
        if args.agent == 'sac':
            agent = SACAgent(input_dim, output_dim, lr=args.lr)
            test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
            args.location = f"savedagents/models/{args.location if args != '' else 'SAC_Model'}"
            agent.agent = SAC.load(args.location if args != '' else 'SAC_Model', env=test_env)
            # agent.agent.load_replay_buffer(f"{args.location}_replay_buffer")
        elif args.agent == 'a2c':
            agent = A2CAgent(input_dim, output_dim, lr=args.lr)
            test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
            args.location = f"savedagents/models/{args.location if args != '' else 'A2C_Model'}"
            agent.agent = A2C.load(args.location if args != '' else 'A2C_Model', env=test_env)
            # agent.agent.load_replay_buffer(f"{args.location}_replay_buffer")
        elif args.agent == 'custom':
            pass
            """agent = CustomAgent(input_dim, output_dim, lr=args.lr)
            test_env.observation_space = agent.convert_observation_space(test_env.observation_space)
            args.location = f"savedagents/models/{args.location if args != '' else 'Custom_Model'}"
            agent.agent = SAC.load(args.location if args != '' else 'Custom_Model', env=test_env)
            # agent.agent.load_replay_buffer(f"{args.location}_replay_buffer")"""
        elif args.agent == 'sr':
            formula = "(x1 * (x2 * -0.99871546) * inv(x3))"
            formula = "((x1 * -0.998) / (x3 / x2))"  # "div(mul(X0, -0.998), div(X2, X1))"
            formula = "((-0.005 - x1) / (x3 / x2))"
            agent = SRAgent(input_dim, output_dim, function=formula)
        else:
            raise ValueError('Agent string {} not recognized'.format(args.agent))
        # agent.load_agent(args.location)

        # test_env = agent.agent.get_env()  # maybe the better solution?

        print(f"Loaded agent from Model {args.agent} from location {args.location}")

        # use the loaded agent
        mean_test_rewards = agent.test_loop(test_env=test_env, config=args, verbose=1)
        # mean_test_rewards = agent.evaluate_model(test_env=test_env, config=args)
    end_time = time.time()
    print(end_time - start_time)
