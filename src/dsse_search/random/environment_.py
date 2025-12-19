import logging
import random
import functools
import itertools
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from gymnasium.spaces import Tuple as GymTuple
from typing import Optional, Tuple
from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss
from dsse_search.random.environment.env import DroneSwarmSearch


def random_person_and_drone_initial_position_environment_creator(grid_size: int, timestep_limit: int, person_amount: int, drone_amount: int,
                        drone_speed: int, detection_probability: float, dispersion_inc: float, render_mode: str = "ansi") -> ParallelPettingZooEnv:

    env = DroneSwarmSearch(
        grid_size=grid_size,
        render_mode=render_mode,
        pre_render_time=0,
        timestep_limit=timestep_limit,
        person_amount=person_amount,
        dispersion_inc=dispersion_inc,
        drone_amount=drone_amount,
        drone_speed=drone_speed,
        probability_of_detection=detection_probability,
        person_initial_position=(grid_size // 2, grid_size // 2),
    )

    env = ParallelPettingZooEnv(env)

    return env

