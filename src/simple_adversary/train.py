import os
import ray
import logging
from ray.tune.registry import register_env
from common.cli import CommonTrainingArgs
from common.ppo import initialize_base_ppo_from_args
from common.tuner import initialize_base_tuner
from argparse_dataclass import dataclass, ArgumentParser
from typing import Literal

from simple_adversary.environment import environment_creator

logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingArgs(CommonTrainingArgs):
    # "group_shared" = Bad agents share a policy, Good agents share a different policy
    mode: Literal["independent", "group_shared"]

    n_good_agents: int
    max_cycles: int
    continuous_actions: bool


def get_policy_config(mode: str) -> dict:
    if mode == "independent":
        return {
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id
        }
    
    elif mode == "group_shared":
        return {
            "policies": {"adversary_policy", "agent_policy"},
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: (
                "adversary_policy" if "adversary" in agent_id else "agent_policy"
            )
        }
    
    raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    args = ArgumentParser(TrainingArgs).parse_args()
    logging.info("Parsed training arguments.")
    logging.info(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logging.info("Initializing Ray...")
    ray.init()

    logging.info("Registering Simple Adversary environment...")
    env_name = "simple_adversary_v3"
    register_env(env_name, lambda config: environment_creator(**config))

    config = (
        initialize_base_ppo_from_args(args)
        .environment(
            env_name,
            env_config={
                "n_good_agents": args.n_good_agents,
                "max_cycles": args.max_cycles,
                "continuous_actions": args.continuous_actions,
            },
        )
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