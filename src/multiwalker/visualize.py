import ray
import logging
from ray.tune.registry import register_env
from eval.visualize import visualize
from multiwalker.environment import environment_creator
from multiwalker.ppo import get_eval_ppo_config
from multiwalker.cli import EvalArgs
from argparse_dataclass import ArgumentParser


logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    args = ArgumentParser(EvalArgs).parse_args()

    logging.info("Initializing Ray...")
    ray.init()

    logging.info("Registering Multiwalker environment...")
    env_name = "multiwalker"
    register_env(env_name, lambda config: environment_creator(**config))
    
    config = get_eval_ppo_config(
        args=args,
        env_name=env_name,
    )

    algo = config.build_algo()

    if args.checkpoint_path:
        logging.info(f"Restoring checkpoint from {args.checkpoint_path}...")
        algo.restore(args.checkpoint_path)

    env = environment_creator(
        n_walkers=args.n_walkers,
        render_mode="human",
    )

    visualize(
        algo,
        env,
        n_episodes=args.n_episodes,
    )

    





