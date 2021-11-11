import numpy as np
from gym.spaces import Box, Discrete, Dict

action_space = Dict({
            "x": Box(low=-10., high=10., shape=(1,1), dtype=np.float32), # x-coordinate to move the box to
            "y": Box(low=1., high=10., shape=(1, 1), dtype=np.float32),  # y-coordinate to move the box to
            "box": Discrete(2) # 1: BoxA, 2: BoxB
        })

for _ in range(10):
    sample = action_space.sample()
    x = sample["x"][0,0]
    y = sample["y"][0, 0]
    box = sample["box"]
    print(x,y,box)