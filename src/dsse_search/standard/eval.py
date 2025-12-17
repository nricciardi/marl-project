import ray
import logging
from evaluate.visualize import visualize
from evaluate.simulate import simulate, plot_simulation_results
from ray.tune.registry import register_env
from dsse_search.standard.environment import environment_creator
from dsse_search.standard.cli import EvalArgs
from argparse_dataclass import ArgumentParser
from ray.rllib.algorithms.algorithm import Algorithm


logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    args = ArgumentParser(EvalArgs).parse_args()

    logging.info("Initializing Ray...")
    ray.init()

    env_name = "dsse_search"
    register_env(env_name, lambda config: environment_creator(**config))
    
    logging.info(f"Restoring checkpoint from {args.checkpoint_path}...")
    algo = Algorithm.from_checkpoint(args.checkpoint_path)

    env = environment_creator(
        render_mode="human",
        grid_size=args.grid_size,
        timestep_limit=args.timestep_limit,
        person_amount=args.person_amount,
        dispersion_inc=args.dispersion_inc,
        drone_amount=args.drone_amount,
        drone_speed=args.drone_speed,
        detection_probability=args.detection_probability,
    )

    results = simulate(
        algo,
        env,
        n_episodes=args.n_episodes,
        explore=args.explore,
    )

    plot_simulation_results(results)

    





