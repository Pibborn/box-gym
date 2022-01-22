import numpy as np
from gym import spaces
import gym
from stable_baselines3 import PPO, A2C, SAC, DQN, DDPG  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Instantiate the env
from ScaleEnvironment.ScaleExperiment import ScaleExperiment

env = ScaleExperiment(rendering=False, randomness=True, actions=1)
low = [env.observation_space[x].low[0] for x in env.observation_space]
high = [env.observation_space[x].high[0] for x in env.observation_space]
env.observation_space = spaces.Box(low=np.array(low), high=np.array(high),
                                   shape=(len(low),), dtype=np.float32)

# wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agentTe
model = SAC('MlpPolicy', env, verbose=1, use_sde=False).learn(50000)
model.save('SAC_Model2')
model = SAC.load('SAC_Model2', env=env)

#model = A2C('MlpPolicy', env, verbose=1, use_sde=True).learn(5000)
#model.save('A2C_Model')
#model = A2C.load('A2C_Model', env=env)


# Test the trained agent
obs = env.reset()
n_steps = 20
episodes = 500
success = 0

#evaluate_policy(model, env, n_eval_episodes=10, render=True)

obs = env.reset()
env = ScaleExperiment(rendering=True, randomness=False, actions=1)
while True:
    for _ in range(100):
        action, states = model.predict(obs)
        state, reward, done, _ = env.step(action)
        """if reward >= 1:
            print(
                    f"end position: {state[1]}   \taction input: {action[0]}    \t{float(str((state[1] - action[0]) / action[0] * 100)[:5])}% difference)")
        """
        if done:
            break

"""for _ in range(episodes):
    obs = env.reset()
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
"""