# box-gym

## 1. Create your virtual environment and install the required packages

```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

## 2. Training the agent 

### Simple testing command

```
python run_agent.py --envname='scale_draw' --episodes=20000 --agent='sac' 
```

### Agent selection
Agents to choose from: [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) (recommended), [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html), [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) (recommended when using raw images as input)
Arguments: ```--agent='sac'```, ```agent='a2c'``` or ```agent='ppo'```

### Saving the agent:
Add the lines ```--overwriting``` and ```--location='path'``` to save the trained agent in the savedagents/models/ folder.

```
python run_agent.py --envname='scale_draw' --episodes=50000 --overwriting --location='SAC_new_agent'
```

### Box setup
Sides: 
* ```--sides=1``` (default): boxes can be placed anywhere on the scale by agent and from environment side when resetting the places 
* ```--sides=2```: the environment chooses places for its boxes on the left side, the agent only is allowed to place his boxes on the right side

Densities:
* ```--random_densities```: densities of boxes can have values between 4 and 6
* else: all boxes have the same density (5.0)

Sizes of the boxes:
* ```--random_boxsizes```: the size of one box is randomized to a value between 0.8 and 1.2
* else: all boxes have size 1.0

Number of boxes:
* ```--actions=2```: number of boxes that have to be placed by the agent (default: 1)
* ```--placed=1```: number of the boxes that have been placed randomly beforehand from the environment (default: 1)


### Other arguments
* Seed: set the seed of an environment individually by adding ```--seeding=1234``` (default: 42)
* If you somehow cannot install Xvfb (like on Mac), add the ```--disable_xvfb``` argument
* Number of episodes: ```--episodes=50000``` (default: 10000)
* Steps between each print: ```--printevery=100``` (default: 500)
* Use pixels as observations instad of extracted values: ```--raw_pixels```
* Normalization of observations and actions: ```--normalize```
* Render the simulation: ```--rendering```

### Example setting
```
python run_agent.py --envname='scale_draw' --random_densities --random_boxsizes --episodes=25000 --overwriting --location="SAC_25000" --raw_pixels
```

## 3. Testing the agent 

* important: use the same box settings (number of boxes, number of actions, ...) and the same location from the training before
* then, run the command by adding the argument ```--test```
* also state how much episodes you want to test the agent (by default: ```--episodes=10000```)
```
python run_agent.py --envname='scale_draw' --random_densities --random_boxsizes --episodes=1000 --location="SAC_25000" --raw_pixels --test
```

## 4. Extracting data from the (pre-trained) agent

...
