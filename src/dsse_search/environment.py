import random
from typing import Optional
from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss
from DSSE import DroneSwarmSearch


def environment_creator(grid_size: int, timestep_limit: int, person_amount: int, drone_amount: int,
                        drone_speed: int, detection_probability: float,
                        dispersion_inc: float, render_mode: str = "ansi") -> ParallelPettingZooEnv:

    speed_vector = (
        random.random() * 2 - 1,
        random.random() * 2 - 1
    )

    person_initial_position = (
        random.randint(0, grid_size - 1),
        random.randint(0, grid_size - 1),
    )

    if person_initial_position[0] < grid_size // 4 and speed_vector[0] < 0:
        speed_vector = (-speed_vector[0], speed_vector[1])
    if person_initial_position[0] > 3 * grid_size // 4 and speed_vector[0] > 0:
        speed_vector = (-speed_vector[0], speed_vector[1])
    if person_initial_position[1] < grid_size // 4 and speed_vector[1] < 0:
        speed_vector = (speed_vector[0], -speed_vector[1])
    if person_initial_position[1] > 3 * grid_size // 4 and speed_vector[1] > 0:
        speed_vector = (speed_vector[0], -speed_vector[1])
    
    env = DroneSwarmSearch(
        grid_size=grid_size,
        render_mode=render_mode,
        pre_render_time=0,
        person_initial_position=person_initial_position,
        vector=speed_vector,
        timestep_limit=timestep_limit,
        person_amount=person_amount,
        dispersion_inc=dispersion_inc,
        drone_amount=drone_amount,
        drone_speed=drone_speed,
        probability_of_detection=detection_probability,
    )

    env = ParallelPettingZooEnv(env)

    return env