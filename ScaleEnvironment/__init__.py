from gym.envs.registration import register

register(
    id='Scale-v0',
    entry_point='ScaleEnvironment:Scale'
)