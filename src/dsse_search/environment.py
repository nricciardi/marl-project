from typing import Optional
from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss
from DSSE import DroneSwarmSearch


def environment_creator(render_mode: Optional[str] = None) -> ParallelPettingZooEnv:
    
    env = DroneSwarmSearch(
        grid_size=40,
        # render_mode="human",
        # render_grid=True,
        # render_gradient=True,
        pre_render_time=0,
        vector=(1, 1),
        timestep_limit=300,
        person_amount=4,
        dispersion_inc=0.05,
        person_initial_position=(15, 15),
        drone_amount=2,
        drone_speed=10,
        probability_of_detection=0.9,
    )

    env = ParallelPettingZooEnv(env)

    return env