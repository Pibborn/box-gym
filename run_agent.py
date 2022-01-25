import time
import matplotlib

from agents.StableBaselinesAgents.A2CAgent import A2CAgent
matplotlib.rcParams['backend'] = 'WebAgg'
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import dill
import matplotlib

matplotlib.rcParams['backend'] = 'WebAgg'
import matplotlib.pyplot as plt

from environments.GymEnv import GymEnv
from ScaleEnvironment.Scale import Scale
from ScaleEnvironment.ScaleExperiment import ScaleExperiment
from agents.StableBaselinesAgents.SACAgent import SACAgent
from gym.spaces import Dict
import argparse
import wandb
from rich.traceback import install
install(show_locals=True)


def create_envs(env_str, seed=42, do_render=True, randomness=False):
    if env_str == 'scale':
        train_env = Scale(rendering=do_render)
        test_env = Scale(rendering=do_render)
    elif env_str == 'scale_exp':
        train_env = ScaleExperiment(rendering=do_render, randomness=randomness, actions=2)
        test_env = ScaleExperiment(rendering=do_render, randomness=randomness, actions=2)
    elif env_str == 'scale_single':
        train_env = ScaleExperiment(rendering=do_render, randomness=randomness, actions=1)
        test_env = ScaleExperiment(rendering=do_render, randomness=randomness, actions=1)
    else:
        train_env = GymEnv(env_str)
        train_env = train_env.create()
        test_env = GymEnv(env_str)
        test_env = test_env.create()
    if seed:
        train_env.seed(seed)
        test_env.seed(seed + 1)
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
    parser.add_argument('envname')
    parser.add_argument('agent', type=str)
    parser.add_argument('--seed', type=int, default=42)                                 # old default: 42
    parser.add_argument('--episodes', type=int, default=10000)                          # old default: 1000
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--printevery', type=int, default=1000)                         # old default: 10
    parser.add_argument('--discount', type=float, default=0.99)                         # old default: 0.99
    parser.add_argument('--threshold', type=float, default=20.1)                        # old default: 475
    parser.add_argument('--dropout', type=float, default=0.2)                           # old default: 0.2
    parser.add_argument('--randomness', action='store_true')                       # old default: False
    parser.add_argument('--rendering', action='store_true')
    parser.add_argument('--overwriting', type=bool, default=False)                      # old default: True
    parser.add_argument('--entity', type=str, default='jgu-wandb')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    wandb.init(project="box-gym", entity=args.entity, config=args, sync_tensorboard=True)

    if not args.test:  # train + test new agent
        train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=False,  # args.rendering,
                                          randomness=args.randomness)  # do_render=True
        input_dim, output_dim = get_env_dims(train_env)
        # agent = QAgent(input_dim, output_dim, gamma=args.discount, lr=args.lr)
        if args.agent == 'sac':
            agent = SACAgent(input_dim, output_dim)
        elif args.agent == 'a2c':
            agent = A2CAgent(input_dim, output_dim)
        else:
            raise ValueError('Agent string {} not recognized'.format(args.agent))
        # agent = A2CAgent(input_dim, output_dim)
        # agent = VanillaGradMLP(input_dim, 100, output_dim, dropout=args.dropout, uses_scale=args.envname=='scale',
        #                     scale_exp=args.envname=='scale_exp')
        agent.train_loop(train_env, test_env, args, verbose=1, only_testing=False)
        # save the trained agent
        if args.overwriting:   # todo: fix pickling
            with open('agent', 'wb') as agent_file:
                dill.dump(agent, agent_file)

    else:  # load old agent and test him
        # load the agent
        with open('agent', 'rb') as agent_file:
            agent = dill.load(agent_file)
            # use the loaded agent
            train_env, test_env = create_envs(args.envname, seed=args.seed, do_render=args.rendering,
                                              randomness=args.randomness)
            _, mean_test_rewards = agent.train_loop(train_env, test_env, args, only_testing=True)

    end_time = time.time()
    print(end_time - start_time)
