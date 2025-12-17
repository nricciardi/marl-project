import os
import ray
import logging
from ray import tune
from ray.tune.registry import register_env
from common.set_seed import set_global_seed
from common.tuner import initialize_base_tuner
from argparse_dataclass import ArgumentParser
from ray.rllib.algorithms.algorithm import Algorithm

from dsse_search.standard.cli import TrainingArgs
from dsse_search.standard.environment import environment_creator
from dsse_search.standard.ppo import get_train_ppo_config


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    args = ArgumentParser(TrainingArgs).parse_args()
    logging.info("Parsed training arguments.")
    logging.info(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logging.info("Initializing Ray...")
    ray.init()

    logging.info("Registering environment...")

    env_name = "standard_dsse_search"
    register_env(env_name, lambda config: environment_creator(**config))
    

    if args.from_checkpoint:
        logging.info(f"Restoring from checkpoint directory {args.from_checkpoint}...")
        algo = Algorithm.from_checkpoint(args.from_checkpoint)
        config = algo.config

        if config is None:
            raise ValueError("Restored algorithm has no config.")

    else:
        logging.info("No checkpoint directory provided, training from scratch.")

        config = get_train_ppo_config(args, env_name)
        
        if args.seed is not None:
            config = config.debugging(seed=args.seed)

            set_global_seed(args.seed)
            logging.info(f"Set global seed to {args.seed}.")
        
        algo = config.build_algo()

    param_space = config.to_dict()

    tuner = initialize_base_tuner(
        args,
        param_space,
        config.algo_class,
    )

    logging.info("Start training...")

    result_grid = tuner.fit()
    
    logging.info("Training completed.")

    ray.shutdown()