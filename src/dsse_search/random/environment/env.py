import functools
import numpy as np
import random
from gymnasium.spaces import MultiDiscrete, Discrete, Box, Tuple
from dsse_search.random.environment.constants import RED, GREEN, Actions, Reward
from dsse_search.random.environment.entities.person import Person
from dsse_search.random.environment.env_base import DroneSwarmSearchBase
from dsse_search.random.environment.simulation.dynamic_probability import ProbabilityMatrix


class DroneSwarmSearch(DroneSwarmSearchBase):
    possible_actions = {action for action in Actions}
    metadata = {
        "name": "DroneSwarmSearchV0",
    }

    reward_scheme = Reward(
        default=-0.15,
        leave_grid=-0.5,
        exceed_timestep=0,
        drones_collision=0,
        search_cell=0,
        search_and_find=40,
        proximity_threshold=0,
    )

    def __init__(
        self,
        grid_size=20,
        render_mode="ansi",
        render_grid=False,
        render_gradient=True,
        vector=(1.1, 1),
        dispersion_inc=0.1,
        dispersion_start=0.5,
        timestep_limit=100,
        person_amount=1,
        person_initial_position=(0, 0),
        drone_amount=1,
        drone_speed=10,
        probability_of_detection=1.0,
        pre_render_time=0,
        grid_cell_size=130,
        fps=5,
    ):
        if person_amount <= 0:
            raise ValueError(f"The number of persons must be greater than 0 (provided: {person_amount}).")
        self.person_amount = person_amount

        super().__init__(
            grid_size=grid_size,
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            timestep_limit=timestep_limit,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            probability_of_detection=probability_of_detection,
            grid_cell_size=grid_cell_size,
            render_fps=fps,
        )

        self.pre_render_steps = round(
            (pre_render_time * 60)
            / (self.calculate_simulation_time_step(drone_speed, self.cell_size))
        )

        self.probability_matrix = None
        self.dispersion_inc = dispersion_inc
        self.dispersion_start = dispersion_start
        self.vector = vector
        self.disaster_position = person_initial_position

        self.person_initial_position = person_initial_position
        self.persons_set = self.create_persons_set()

        self.rewards_sum = {a: 0 for a in self.possible_agents}
        self.rewards_sum["total"] = 0

        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }

    def create_persons_set(self) -> set[Person]:
        persons_set = set()
        position = self.create_random_positions_person(
            self.person_initial_position, self.person_amount
        )

        for i in range(self.person_amount):
            person = Person(
                index=i,
                initial_position=position[i],
                grid_size=self.grid_size,
            )
            person.calculate_movement_vector(self.vector)
            persons_set.add(person)

            if not self.is_valid_position(person.initial_position):
                raise ValueError(f"Person initial position is out of the matrix (provided: {person.initial_position})")

        return persons_set

    def create_random_positions_person(
        self, central_position: tuple[int, int], amount: int, max_distance: int = 2
    ) -> list[tuple[int, int]]:
        if not self.is_valid_position(central_position):
            raise ValueError(f"Central position is out of the matrix (provided: {central_position})")

        max_distance_range = (max_distance * 2 + 1) ** 2

        if amount > max_distance_range:
            raise ValueError(
                f"There are more persons than grid spots. Reduce number of persons or increase grid size (provided: {amount})."
            )

        unique_random_positions = {central_position}
        while len(unique_random_positions) < amount:
            dx = randint(-max_distance, max_distance)
            dy = randint(-max_distance, max_distance)

            if (dx, dy) != (0, 0):
                new_position = (central_position[0] + dx, central_position[1] + dy)
                if self.is_valid_position(new_position):
                    unique_random_positions.add(new_position)

        return list(unique_random_positions)

    def render(self):
        self.pygame_renderer.render_map()
        self.pygame_renderer.render_entities(self.persons_set)
        self.pygame_renderer.render_entities(self.agents_positions)
        self.pygame_renderer.refresh_screen()

    def reset(
        self,
        seed=None,
        options=None,
    ):
        
        if options is not None:
            print(f"Warning: Options passed to reset() will be ignored in this environment. Options received: {options}")

        # self.person_amount = random.randint(1, 3)
        self.person_initial_position = (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        )

        self.disaster_position = self.person_initial_position

        speed_vector = (
            random.random() * 2 - 1,
            random.random() * 2 - 1
        )

        # print(f"New person initial position: {self.person_initial_position}, speed vector: {speed_vector}")

        if self.person_initial_position[0] < self.grid_size // 4 and speed_vector[0] < 0:
            speed_vector = (-speed_vector[0], speed_vector[1])
        if self.person_initial_position[0] > 3 * self.grid_size // 4 and speed_vector[0] > 0:
            speed_vector = (-speed_vector[0], speed_vector[1])
        if self.person_initial_position[1] < self.grid_size // 4 and speed_vector[1] < 0:
            speed_vector = (speed_vector[0], -speed_vector[1])
        if self.person_initial_position[1] > 3 * self.grid_size // 4 and speed_vector[1] > 0:
            speed_vector = (speed_vector[0], -speed_vector[1])

        drones_positions = []
        while len(drones_positions) < self.drone.amount:
            new_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if new_position not in drones_positions:
                drones_positions.append(new_position)

        # print(f"New drones initial positions: {drones_positions}")

        options = {
            "drones_positions": drones_positions,
            "vector": speed_vector,
        }

        vector = options.get("vector") if options else None
        self.vector = vector if vector else self.vector

        self.persons_set = self.create_persons_set()
        for person in self.persons_set:
            person.reset_position()
            person.update_time_step_relation(self.time_step_relation, self.cell_size)

        pod_multiplier = options.get("person_pod_multipliers") if options else None

        if pod_multiplier is not None:
            self.raise_if_unvalid_mult(pod_multiplier)
            for person, mult in zip(self.persons_set, pod_multiplier):
                person.set_mult(mult)

        self.probability_matrix = ProbabilityMatrix(
            40,
            self.dispersion_start,
            self.dispersion_inc,
            self.vector,
            [
                self.disaster_position[1],
                self.disaster_position[0],
            ],
            self.grid_size,
        )

        self.probability_matrix.update_time_step_relation(
            self.time_step_relation, self.cell_size
        )

        observations, infos = super().reset(seed=seed, options=options)
        self.rewards_sum = {a: 0 for a in self.agents}
        self.rewards_sum["total"] = 0
        return observations, infos

    def raise_if_unvalid_mult(self, individual_multiplication: list[int]) -> bool:
        if len(individual_multiplication) != len(self.persons_set):
            raise ValueError(
                f"The number of multipliers must be equal to the number of persons (provided: {len(individual_multiplication)})."
            )

        for mult in individual_multiplication:
            if not isinstance(mult, (int, float)) or mult < 0:
                raise ValueError("The multiplier must be a positive number.")

    def pre_search_simulate(self):
        for _ in range(self.pre_render_steps):
            self.create_observations()
            if self.render_mode == "human":
                self.render()

    def create_observations(self):
        observations = {}

        for person in self.persons_set:
            movement_map = self.build_movement_matrix(person)
            person.step(movement_map)

        self.probability_matrix.step()
        probability_matrix = self.probability_matrix.get_matrix().astype(np.float32)

        for idx, agent in enumerate(self.agents):
            current_pos = np.array(self.agents_positions[idx])
            
            others_with_dist = []
            for i, pos in enumerate(self.agents_positions):
                if i != idx:
                    other_pos = np.array(pos)
                    distance = np.linalg.norm(current_pos - other_pos)
                    others_with_dist.append((other_pos, distance))
            
            others_with_dist.sort(key=lambda x: x[1])
            
            ordered_coords = [current_pos] + [item[0] for item in others_with_dist]
            flattened_coords = np.concatenate(ordered_coords).astype(np.float32)
            
            observations[agent] = (
                flattened_coords,
                probability_matrix,
            )

        return observations

    def step(self, actions):
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")

        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme.default for a in self.agents}
        truncations = {a: False for a in self.agents}
        person_found = False

        probability_matrix = self.probability_matrix.get_matrix()

        for idx, agent in enumerate(self.agents):
            if agent not in actions:
                raise ValueError(f"Missing action for {agent}")

            drone_action = actions[agent]
            
            if self.timestep >= self.timestep_limit:
                rewards[agent] = self.reward_scheme.exceed_timestep
                truncations[agent] = True
                terminations[agent] = True
                continue

            drone_x, drone_y = self.agents_positions[idx]
            is_searching = drone_action == Actions.SEARCH.value

            if drone_action != Actions.SEARCH.value:
                new_position = self.move_drone((drone_x, drone_y), drone_action)
                if not self.is_valid_position(new_position):
                    rewards[agent] = self.reward_scheme.leave_grid
                else:
                    self.agents_positions[idx] = new_position
                    rewards[agent] = self.reward_scheme.default
            else:
                drone_found_person = False
                for human in self.persons_set:
                    drone_found_person = (
                        human.x == drone_x and human.y == drone_y and is_searching
                    )
                    if drone_found_person:
                        break

                if drone_found_person:
                    max_detection_probability = min(human.get_mult() * self.drone.pod, 1)
                    if random.random() <= max_detection_probability:
                        self.persons_set.remove(human)
                        time_decay_factor = 1 - (self.timestep / self.timestep_limit)
                        rewards[agent] = self.reward_scheme.search_and_find + (self.reward_scheme.search_and_find * time_decay_factor)

                if len(self.persons_set) == 0:
                    person_found = True
                    for a in self.agents:
                        terminations[a] = True
                        truncations[a] = True

            current_cell_prob = probability_matrix[self.agents_positions[idx][1], self.agents_positions[idx][0]]
            if current_cell_prob > self.reward_scheme.proximity_threshold:
                rewards[agent] += current_cell_prob

            self.rewards_sum[agent] += rewards[agent]

        self.timestep += 1
        infos = {drone: {"Found": person_found} for drone in self.agents}
        self.render_step(any(terminations.values()), person_found)
        observations = self.create_observations()
        
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
            
        return observations, rewards, terminations, truncations, infos

    def render_step(self, terminal, person_found):
        if self.render_mode == "human":
            if terminal:
                if person_found:
                    self.pygame_renderer.render_episode_end_screen(
                        f"The target was found in {self.timestep} moves", GREEN
                    )
                else:
                    self.pygame_renderer.render_episode_end_screen(
                        "The target was not found.", RED
                    )
            else:
                self.render()

    def build_movement_matrix(self, person: Person) -> np.ndarray:
        left_x = max(person.x - 1, 0)
        right_x = min(person.x + 2, self.grid_size)
        left_y = max(person.y - 1, 0)
        right_y = min(person.y + 2, self.grid_size)

        probability_matrix = self.probability_matrix.get_matrix()
        movement_map = probability_matrix[left_y:right_y, left_x:right_x].copy()

        if person.x == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=1)
        elif person.x == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=1)

        if person.y == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=0)
        elif person.y == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=0)

        return movement_map

    def get_persons(self):
        return self.persons_set

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Tuple(
            (
                MultiDiscrete([self.grid_size] * (self.drone.amount * 2)),
                Box(
                    low=0,
                    high=1,
                    shape=(self.grid_size, self.grid_size),
                    dtype=np.float32,
                ),
            )
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.possible_actions))