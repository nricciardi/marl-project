from typing import Optional
from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss


def environment_creator(n_walkers: int, parallel: bool, stacked_frames: int, render_mode: Optional[str] = None) -> PettingZooEnv | ParallelPettingZooEnv:
    
    if parallel:
        env = multiwalker_v9.parallel_env(n_walkers=n_walkers, render_mode=render_mode)
    else:
        env = multiwalker_v9.env(n_walkers=n_walkers, render_mode=render_mode)

    env = ss.frame_stack_v1(env, stack_size=stacked_frames)

    if parallel:
        return ParallelPettingZooEnv(env)
    else:
        return PettingZooEnv(env)