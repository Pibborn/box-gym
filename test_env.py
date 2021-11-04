#from ScaleEnvironment.Scale import Scale
from ScaleEnvironment.ScaleFixedPositions import Scale
from ScaleEnvironment.framework import main

if __name__ == '__main__':
    env = Scale
    main(Scale)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
