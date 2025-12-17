import random
from typing import Optional, Tuple
from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss
from DSSE import DroneSwarmSearch


def standard_environment_creator(grid_size: int, timestep_limit: int, person_amount: int, drone_amount: int,
                        drone_speed: int, detection_probability: float, person_speed: Tuple[float, float], 
                        person_initial_position: Tuple[int, int], dispersion_inc: float, render_mode: str = "ansi") -> ParallelPettingZooEnv:

    env = DroneSwarmSearch(
        grid_size=grid_size,
        render_mode=render_mode,
        pre_render_time=0,
        person_initial_position=person_initial_position,
        vector=person_speed,
        timestep_limit=timestep_limit,
        person_amount=person_amount,
        dispersion_inc=dispersion_inc,
        drone_amount=drone_amount,
        drone_speed=drone_speed,
        probability_of_detection=detection_probability,
    )

    env = ParallelPettingZooEnv(env)

    return env

