import os
import ray
import logging
from ray.tune.registry import register_env
from common.cli import CommonTrainingArgs
from common.ppo import initialize_base_training_ppo_from_args
from common.tuner import initialize_base_tuner
from argparse_dataclass import dataclass, ArgumentParser
from typing import Literal

from multiwalker.environment import environment_creator

logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingArgs(CommonTrainingArgs):
    # "group_shared" = Bad agents share a policy, Good agents share a different policy
    mode: Literal["independent", "shared"]

    n_walkers: int


def get_policy_config(mode: str, n_walkers: int) -> dict:
    
    if mode == "shared":
        return {
            "policies": {
                "shared_policy"
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy"
        }
    
    elif mode == "independent":
        return {
            "policies": set(f"agent_{i}_policy" for i in range(n_walkers)),
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

    logging.info("Registering Simple Tag environment...")
    env_name = "simple_tag"
    register_env(env_name, lambda config: environment_creator(**config))

    config = (
        initialize_base_training_ppo_from_args(args)
        .environment(
            env_name,
            env_config={
                "n_walkers": args.n_walkers,
            },
        )
        .multi_agent(
            **get_policy_config(args.mode, args.n_walkers)
        )
        # .training(
        #     model={
        #         "fcnet_hiddens": [256, 256],
        #         "fcnet_activation": "tanh",
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