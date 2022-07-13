import argparse

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str, default='scale_single')
    parser.add_argument('--agent', type=str, default='sac')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--printevery', type=int, default=500)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--entity', type=str, default='jgu-wandb')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=20.1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rendering', action='store_true')
    parser.add_argument('--overwriting', action='store_true')
    parser.add_argument('--location', type=str, default="")
    parser.add_argument('--reward-norm', action='store_true')
    parser.add_argument('--disable_xvfb', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--path', type=str, default='results')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--mode', type=int, default=1)
    # scale settings
    parser.add_argument('--random_densities', action='store_true')
    parser.add_argument('--random_boxsizes', action='store_true')
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
    # additional orbit settings
    parser.add_argument('--random_planet_position', action='store_true')
    parser.add_argument('--random_gravity', action='store_true')
    parser.add_argument('--random_satellite_size', action='store_true')
    parser.add_argument('--random_satellite_position', action='store_true')
    parser.add_argument('--random_satellite_density', action='store_true')
    args = parser.parse_args()
    return args