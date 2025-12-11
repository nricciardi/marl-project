import os
import ray
import logging
from ray.tune.registry import register_env
from common.tuner import initialize_base_tuner
from argparse_dataclass import ArgumentParser
from multiwalker.cli import TrainingArgs
from multiwalker.environment import environment_creator
from multiwalker.ppo import get_train_ppo_config


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    args = ArgumentParser(TrainingArgs).parse_args()
    logging.info("Parsed training arguments.")
    logging.info(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logging.info("Initializing Ray...")
    ray.init()

    logging.info("Registering Multiwalker environment...")
    env_name = "multiwalker"
    register_env(env_name, lambda config: environment_creator(**config))

    config = (get_train_ppo_config(args, env_name)
        # .training(
        #     model={
        #         "fcnet_hiddens": [256, 256],
        #         "fcnet_activation": "relu",
        #     }
        # )
    )
    
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