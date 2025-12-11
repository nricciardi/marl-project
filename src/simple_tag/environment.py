from mpe2 import simple_tag_v3
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


def environment_creator(n_good_agents: int, n_bad_agents: int, n_obstacles: int, max_cycles: int, continuous_actions: bool):
    
    env = simple_tag_v3.env(
        num_good=n_good_agents,
        num_adversaries=n_bad_agents, 
        num_obstacles=n_obstacles, 
        max_cycles=max_cycles, 
        continuous_actions=continuous_actions
    )

    return PettingZooEnv(env)