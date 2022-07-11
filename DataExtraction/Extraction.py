from time import sleep

import pandas as pd
import numpy as np

from ScaleEnvironment.ScaleDraw import rescale_movement


def init_Scale(box_number=2, save_boxsize=True):
    # create DataFrame: first 1/3 columns are positions, the next 1/3 densities and the rest the sizes of the boxes
    df_dict = {}
    df_dict.update({f'Position {i + 1}': pd.Series(dtype='float') for i in range(box_number)})
    df_dict.update({f'Density {i + 1}': pd.Series(dtype='float') for i in range(box_number)})
    if save_boxsize:
        df_dict.update({f'Boxsize {i + 1}': pd.Series(dtype='float') for i in range(box_number)})
    df = pd.DataFrame(df_dict)
    return df


def init_Basketball():
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
    return df


def init_Orbit():
    df_dict = {}
    # todo
    df = pd.DataFrame(df_dict)
    return df


def update_Scale_table(df, state, config, box_number=2, index=0):
    # index = len(df) # ???
    density_index = 2 + box_number
    size_index = 2 + 2 * box_number
    if config.normalize:
        # only can access positions of placed boxes here
        positions = rescale_movement([-1, 1], state[:density_index - 2], [-20, 20])
        densities = rescale_movement([0, 1], state[density_index:size_index], [4, 6])
        sizes = rescale_movement([0, 1], state[size_index:], [0.8, 1.2])
    else:
        positions = state[:density_index - 2]
        densities = state[density_index:size_index]
        sizes = state[size_index:]
    df.loc[index] = np.concatenate((positions, densities, sizes))
    return df


def update_Basketball_table(df, state, env, config, start_position_x, start_position_y, action, index=0):
    # unwrap DummyVecEnv
    world_width = env.get_attr("world_width")
    world_height = env.get_attr("world_height")
    if config.normalize:  # todo: solve problems with unknown name (?)
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
                                  np.array([world_width, world_height, 15, 15, 1.5, 6,
                                            world_width, world_height, 3])])

    df.loc[index] = np.array(state)
    return df


def update_Orbit_table(df, index=0):
    # todo
    return df