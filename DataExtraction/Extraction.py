import math
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
    df = pd.DataFrame({'x-Position Start': pd.Series(dtype='float'),
                       'y-Position Start': pd.Series(dtype='float'),
                       'Force vector x': pd.Series(dtype='float'),
                       'Force vector y': pd.Series(dtype='float'),
                       'Radius': pd.Series(dtype='float'),
                       'Density': pd.Series(dtype='float'),
                       'x-Position End': pd.Series(dtype='float'),
                       'y-Position End': pd.Series(dtype='float'),
                       'Velocity x': pd.Series(dtype='float'),
                       'Velocity y': pd.Series(dtype='float'),
                       'Gravity': pd.Series(dtype='float'),
                       'Time': pd.Series(dtype='float'),
                       })
    return df


def init_Orbit():
    df_dict = {}
    # todo
    df = pd.DataFrame(df_dict)
    return df

def init_FreeFall():
    # data frame for free fall experiment
    df = pd.DataFrame({'Start Distance': pd.Series(dtype='float'),
                       #'Height': pd.Series(dtype='float'),
                       'Distance': pd.Series(dtype='float'),
                       'Velocity': pd.Series(dtype='float'),
                       'Radius': pd.Series(dtype='float'),
                       'Density': pd.Series(dtype='float'),
                       'Time': pd.Series(dtype='float'),
                       'Gravity': pd.Series(dtype='float'),
                       'Prediction': pd.Series(dtype='float'),
                       'Calculation1': pd.Series(dtype='float'),
                       'Calculation2': pd.Series(dtype='float')
                       })
    return df


def update_Scale_table(df, state, env, config, box_number=2, index=0, action=None):
    # index = len(df) # ???
    #state = env.env_method("resetState")[0]
    density_index = 2 + box_number
    size_index = 2 + 2 * box_number
    if config.normalize:
        # only can access positions of placed boxes here
        positions = rescale_movement([-1, 1], state[:density_index - 2], [-20, 20])
        #positions_placed = rescale_movement([-1, 1], state[:density_index - 2 - config.placed], [-20, 20])
        #positions_action = rescale_movement([-1, 1], action[0], [-20, 20])
        densities = rescale_movement([0, 1], state[density_index:size_index], [4, 6])
        sizes = rescale_movement([0, 1], state[size_index:], [0.8, 1.2])
    else:
        positions = state[:density_index - 2]
        #positions_placed = state[:density_index - 2 - config.placed]
        #positions_action = action[0]
        densities = state[density_index:size_index]
        sizes = state[size_index:]
    df.loc[index] = np.concatenate((positions, densities, sizes))
    #df.loc[index] = np.concatenate((positions_placed, positions_action, densities, sizes))
    return df


def update_Basketball_table(df, state, env, config, start_position_x, start_position_y, action, index=0):
    # unwrap DummyVecEnv
    world_width = env.get_attr("world_width")
    world_height = env.get_attr("world_height")

    #time = env.get_attr("time_passed")[0]
    x_start, y_start, _, _, _, r, den, _, _, _, g, _ = env.get_attr("first_state")[0]
    x_end, y_end, _, v_x, v_y, _, _, _, _, _, _, time = env.get_attr("last_state")[0]
    t = time / 60

    """if config.normalize:  # todo: solve problems with unknown name (?)
        #state = rescale_movement([np.array([0, 0, -1, -1, 0, 0, 0, 0, 0]), np.array([1 for _ in range(9)])],
        #                         state,
        #                         [np.array([0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5]),
        #                          np.array([env.world_width, env.world_height, np.pi, 10, 1.5, 6,
        #                                    env.world_width, env.world_height, 3])])
        # rescaled_action = rescale_movement([0, 1], action, [0, 15])
        state = rescale_movement([np.array([0, 0, -1, -1, 0, 0, 0, 0, 0]), np.array([1 for _ in range(9)])],
                                 np.concatenate(
                                     (np.array([start_position_x, start_position_y]), action[0], state[4:])),
                                 [np.array([0, 0, 0, 0, 0.5, 4, 0, 0, 0.5]),
                                  np.array([world_width, world_height, 15, 15, 1.5, 6,
                                            world_width, world_height, 3])])"""
    #print(state)
    #print(action)
    #ball = env.get_attr("ball")[0]
    #print(ball.ball.linearVelocity)


    """print(env.get_attr("first_state")[0])
    print(env.get_attr("last_state")[0])
    print(env.get_attr("final_state")[0])
    print("_____________________________________")
    print(state)
    state = np.concatenate((np.array([start_position_x, start_position_y]), action[0], state[4:]))
    print(state)
    print(time)

    v0 = action[0][1]

    g = env.get_attr("first_state")[0][-1]
    print(v0, t, g)
    print("errechnet", v0 - g * t)
    print("beobachtet", env.get_attr("final_state")[0][4])
    print("theoretische Zeit", math.sqrt((env.get_attr("final_state")[0][4] + g) / v0))"""

    entry = [x_start, y_start, action[0][0], action[0][1], r, den, x_end, y_end, v_x, v_y, g, t]
    #print(entry)
    #df.loc[index] = np.append(state, time)
    df.loc[index] = np.array(entry)
    return df


def update_Orbit_table(df, index=0):
    # todo
    return df

def update_FreeFall_table(df, state, env, config, start_distance, action=None, max_time=300, index=0):
    #world_height = env.get_attr("world_height")
    # print(state)
    world_height = env.world_height
    use_seconds = env.use_seconds
    velocity = env.last_velocity
    FPS = 60
    if config.normalize:
        """state = rescale_movement([np.array([0, 0, -1, -1, 0, 0, 0, 0, 0]), np.array([1 for _ in range(9)])],
                                 state,
                                 [np.array([0, 0, - np.pi, -10, 0.5, 4, 0, 0, 0.5]),
                                  np.array([env.world_width, env.world_height, np.pi, 10, 1.5, 6,
                                            env.world_width, env.world_height, 3])])"""
        # rescaled_action = rescale_movement([0, 1], action, [0, 15])
        state = rescale_movement(
            # [self.observation_space.low, self.observation_space.high],
            np.array([[0, 0, 0.5, 4, 0, 3],
                      [world_height, 10, 1.5, 6, max_time, 20]]),
            state,
            np.array([np.array([0 for _ in range(6)]), np.array([1 for _ in range(6)])]))
    if not use_seconds:
        state[4] /= FPS
    state[1] = abs(velocity)
    g = state[-1]
    v_t = state[1]
    t = state[-2]
    s = state[0]
    #print(s, g, t)
    #print(v_t, (g * (t - 2/60)))
    #sleep(1)
    #t_1 = math.sqrt(start_distance / (0.50643056 * g))
    t_1 = math.sqrt(start_distance / 4.879745)
    t_2 = (v_t + 0.2123496) / (g * (0.9659417))
    g_x = 0.31125173 + 0.49585986 * g * t ** 2
    print(t_1, t, action[0]/FPS)
    #df.loc[index] = np.append([start_distance], state)
    df.loc[index] = np.append([start_distance], np.append(state, [action[0]/FPS, t_1, t_2]))
    return df