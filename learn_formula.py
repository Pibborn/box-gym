import time

from ArgumentParser import create_argparser
from agents.OtherAgents.SRAgent import SRAgent
from agents.StableBaselinesAgents.A2CAgent import A2CAgent
from agents.StableBaselinesAgents.PPOAgent import PPOAgent
from agents.StableBaselinesAgents.SACAgent import SACAgent
from agents.StableBaselinesAgents.HERAgent import HERAgent
from stable_baselines3.sac import SAC
from stable_baselines3.a2c import A2C

from environments.GymEnv import GymEnv
from ScaleEnvironment.Scale import Scale
from ScaleEnvironment.ScaleExperiment import ScaleExperiment
from ScaleEnvironment.ScaleDraw import ScaleDraw
from environments.BasketballEnvironment import BasketballEnvironment
from environments.OrbitEnvironment import OrbitEnvironment
from DataExtraction.WandB import wandbCSVTracking

import wandb
from gym.spaces import Dict
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from rich.traceback import install

from run_agent import create_envs, get_env_dims, create_agent

install(show_locals=True)


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
                                          sides=args.sides, raw_pixels=args.raw_pixels, walls=args.walls,
                                          random_density=args.random_density, random_ball_size=args.random_ball_size,
                                          random_basket=args.random_basket,
                                          random_ball_position=args.random_ball_position,
                                          random_gravity=args.random_gravity,
                                          random_satellite_position=args.random_satellite_position,
                                          random_planet_position=args.random_planet_position,
                                          random_satellite_size=args.random_satellite_size,
                                          random_satellite_density=args.random_satellite_size)
        input_dim, output_dim = get_env_dims(train_env)

        agent, args.location = create_agent(agentname=args.agent, location=args.location, args=args)

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
                                          sides=args.sides)
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

    # now do the symbolic regression
    # todo: connect the julia script to this point

    end_time = time.time()
    print(end_time - start_time)

    # Finish the run (useful in notebooks)
    run.finish()
