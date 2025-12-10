import supersuit as ss
from pettingzoo.butterfly import cooperative_pong_v5
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


def environment_creator(render_mode="human"):
    """
    Must match the training preprocessing exactly.
    """
    env = cooperative_pong_v5.env(render_mode=render_mode)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.dtype_v0(env, "float32")
    return PettingZooEnv(env)