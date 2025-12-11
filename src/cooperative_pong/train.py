import os
import ray
import logging
from ray.tune.registry import register_env
from common.cli import CommonTrainingArgs
from common.ppo import initialize_base_training_ppo_from_args
from common.tuner import initialize_base_tuner
from cooperative_pong.environment import environment_creator
from argparse_dataclass import dataclass, ArgumentParser
from typing import Literal


logging.basicConfig(level=logging.INFO)


@dataclass
class TrainingArgs(CommonTrainingArgs):
    mode: Literal["shared", "independent"]


def get_policy_config(mode: str) -> dict:
    
    if mode == "shared":
        return {
            "policies": {
                "shared_policy"
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy"
        }
    
    elif mode == "independent":
        return {
            "policies": {
                "agent_0_policy",
                "agent_1_policy"
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: f"agent_{agent_id}_policy"
        }
    
    raise ValueError(f"Unknown mode: {mode}")

    
if __name__ == "__main__":
    args = ArgumentParser(TrainingArgs).parse_args()
    logging.info("Parsed training arguments.")
    logging.info(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logging.info("Initializing Ray...")
    ray.init()

    logging.info("Registering Cooperative Pong environment...")
    env_name = "cooperative_pong_v5"
    register_env(env_name, environment_creator)

    config = (
        initialize_base_training_ppo_from_args(args)
        .environment(env_name)
        .multi_agent(
            **get_policy_config(args.mode)
        )
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
    