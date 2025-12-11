from mpe2 import simple_adversary_v3 
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


def environment_creator(n_good_agents: int, max_cycles: int, continuous_actions: bool):
    
    env = simple_adversary_v3.env(
        N=n_good_agents,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
    )

    return PettingZooEnv(env)