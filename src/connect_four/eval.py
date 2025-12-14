import ray
import logging
from evaluate.visualize import visualize
from evaluate.simulate import simulate, plot_simulation_results
from ray.tune.registry import register_env
from argparse_dataclass import ArgumentParser
from ray.rllib.algorithms.algorithm import Algorithm

from connect_four.environment import environment_creator
from connect_four.cli import EvalArgs

logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    args = ArgumentParser(EvalArgs).parse_args()

    logging.info("Initializing Ray...")
    ray.init()

    env_name = "connect_four"
    register_env(env_name, lambda config: environment_creator(**config))
    
    logging.info(f"Restoring checkpoint from {args.checkpoint_path}...")
    algo = Algorithm.from_checkpoint(args.checkpoint_path)

    env = environment_creator(
        render_mode="human",
    )

    results = simulate(
        algo,
        env,
        n_episodes=args.n_episodes,
        explore=args.explore,
        sleep_time=args.sleep_time
    )

    plot_simulation_results(results)

    





