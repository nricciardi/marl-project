from typing import Optional
from vmas import make_env, Wrapper
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss
import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from vmas import make_env



def environment_creator(n_agents: int, wall_length: float, agent_radius: float, agent_spacing: float, ball_radius: float, stacked_frames: int, continuous_actions: bool, render_mode: Optional[str] = None):
    
    env = make_env(
        scenario="buzz_wire",
        num_envs=1,
        n_agents=n_agents,
        wall_length=wall_length,
        agent_radius=agent_radius,
        agent_spacing=agent_spacing,
        ball_radius=ball_radius,
        render_mode=render_mode,
        continuous_actions=continuous_actions,
        wrapper=Wrapper.RLLIB,
    )

    env = ss.frame_stack_v1(env, stack_size=stacked_frames)

    return env