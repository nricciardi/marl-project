from typing import Optional
from pettingzoo.classic import connect_four_v3
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss


def environment_creator(render_mode: Optional[str] = None) -> PettingZooEnv:
    
    env = connect_four_v3.env(render_mode=render_mode)
    
    # env = aec_to_parallel(env)

    # env = ss.flatten_v0(env)
    
    return PettingZooEnv(env)
