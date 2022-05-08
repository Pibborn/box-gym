# Basketball environment

## 1. Create your virtual environment and install the required packages

```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

## 2. Training the agent 
For documentation for the box environment, click [here](README.md)

### Simple testing command

```
python run_agent.py --envname='basketball' --episodes=20000 --agent='sac' 
```

### Agent selection
Agents to choose from: [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) (recommended), [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html), [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) (recommended when using raw images as input)

Arguments: ```--agent='sac'```, ```agent='a2c'``` or ```agent='ppo'```

### Saving the agent:
Add the lines ```--overwriting``` and ```--location='path'``` to save the trained agent in the savedagents/models/ folder.

```
python run_agent.py --envname='basketball' --episodes=50000 --overwriting --location='SAC_new_agent'
```

### Basketball setup
Density:
* ```--random_density```: density of the ball can have values between 4 and 6
* else: ball has density 5.0

Random Basket:
* ```--random_basket```: basket has a random width, is located randomly on the right side of the window and has a random size/radius
* else: radius is 1.8, width of the basket is 0.4, basket is located at a height equivalent to 60% of the window height (default: 60% of 30) 

Random ball position:
* ```--random_ball_position```: the starting position of the ball is random 
  * x value between 10% and 50% of the width of the world coordinates (here: between 10% - 50% of 40)
  * y value between 40% and 70% of the height of the world coordinates (here: between 40% - 70% of 30)
* else: 
  * start x position: 40% of width
  * start y position: 60% of height

Random ball size:
* ```--random_ball_size```: the radius of the ball is randomized to a value between 0.5 and 1.5
* else: the ball's radius is 1.0

Walls: 
* ```--walls=0``` (default): there is no wall at all (recommended)
* ```--walls=1```: there is a wall on the right side of the window
* ```--walls=2```: there is both a wall on the right and on the left side of the window

### Other arguments
* Seed: set the seed of an environment individually by adding ```--seeding=1234``` (default: 42)
* If you somehow cannot install Xvfb (like on Mac), add the ```--disable_xvfb``` argument
* Number of episodes: ```--episodes=50000``` (default: 10000)
* Steps between each print: ```--printevery=100``` (default: 500)
* Use pixels as observations instead of extracted values: ```--raw_pixels```
* Normalization of observations and actions: ```--normalize```
* Render the simulation: ```--rendering```

### Example setting
```
python run_agent.py --envname='basketball' --random_density --random_ball_size --random_ball_position --episodes=25000 --overwriting --location="SAC_25000" --raw_pixels
```

## 3. Testing the agent 

* important: use the same basketball settings (random ball radius, random start position, ...) and the same location from the training before
* then, run the command by adding the argument ```--test```
* also state how much episodes you want to test the agent (by default: ```--episodes=10000```)
```
python run_agent.py --envname='basketball' --random_density --random_ball_size --random_ball_position --episodes=1000 --location="SAC_25000" --raw_pixels --test
```

## 4. Extracting data from the (pre-trained) agent

...
