import random
from typing import Optional, Tuple
from pettingzoo.sisl import multiwalker_v9
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import supersuit as ss
from DSSE import DroneSwarmSearch


def environment_creator(grid_size: int, timestep_limit: int, person_amount: int, drone_amount: int,
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


class _RandomPersonAndDroneInitialPositionDroneSwarmSearchEnvironment(DroneSwarmSearch):

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        if options is not None:
            print("Warning: options passed to reset() will be ignored.")

        # self.person_amount = random.randint(1, 3)
        self.person_initial_position = (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        )

        speed_vector = (
            random.random() * 2 - 1,
            random.random() * 2 - 1
        )

        if self.person_initial_position[0] < self.grid_size // 4 and speed_vector[0] < 0:
            speed_vector = (-speed_vector[0], speed_vector[1])
        if self.person_initial_position[0] > 3 * self.grid_size // 4 and speed_vector[0] > 0:
            speed_vector = (-speed_vector[0], speed_vector[1])
        if self.person_initial_position[1] < self.grid_size // 4 and speed_vector[1] < 0:
            speed_vector = (speed_vector[0], -speed_vector[1])
        if self.person_initial_position[1] > 3 * self.grid_size // 4 and speed_vector[1] > 0:
            speed_vector = (speed_vector[0], -speed_vector[1])

        drones_positions = [
            (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            for _ in range(self.drone.amount)
        ]

        opt = {
            "drones_positions": drones_positions,
            "vector": speed_vector,
        }

        return super().reset(options=opt, seed=seed)



def random_person_and_drone_initial_position_environment_creator(grid_size: int, timestep_limit: int, person_amount: int, drone_amount: int,
                        drone_speed: int, detection_probability: float, person_speed: Tuple[float, float], 
                        person_initial_position: Tuple[int, int], dispersion_inc: float, render_mode: str = "ansi") -> ParallelPettingZooEnv:

    env = _RandomPersonAndDroneInitialPositionDroneSwarmSearchEnvironment(
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