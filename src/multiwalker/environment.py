from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


def environment_creator(n_walkers: int):
    
    env = multiwalker_v9.env(n_walkers=n_walkers)

    return PettingZooEnv(env)