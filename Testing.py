import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C, SAC, DQN, DDPG  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env

# Instantiate the env
from ScaleEnvironment.ScaleExperiment import ScaleExperiment

env = ScaleExperiment(rendering=False, randomness=False, actions=1)
low = [env.observation_space[x].low[0] for x in env.observation_space]
high = [env.observation_space[x].high[0] for x in env.observation_space]
env.observation_space = spaces.Box(low=np.array(low), high=np.array(high),
                                   shape=(len(low),), dtype=np.float32)

# wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agentTe
model = SAC('MlpPolicy', env, verbose=1).learn(5000)
#model = DDPG('MlpPolicy', env, verbose=1).learn(5000)

# Test the trained agent
obs = env.reset()
n_steps = 20
episodes = 500
success = 0
for _ in range(episodes):
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render(mode='console')
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            if reward >= 1:
                success += 1
            break

print(f"{success} / {episodes} = {success / episodes}")
