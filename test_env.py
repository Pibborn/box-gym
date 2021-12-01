#from ScaleEnvironment.Scale import Scale
from time import sleep

from ScaleEnvironment.Scale import Scale
from ScaleEnvironment.framework import main

if __name__ == '__main__':
    env = Scale()
    #main(Scale)

    for i_episode in range(1000):
        observation = env.reset()
        for t in range(100):
            action = env.action_space.sample()
            if t%10 != 0:
                action = None
            #action = None
            observation, reward, done, info = env.step(action=action)
            env.render()
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()





    """
    env = Scale()
    episodes = 100000
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            print(env.renderer)
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print(n_state[0].angle)
        print('Episode:{} Score:{}'.format(episode, score))"""