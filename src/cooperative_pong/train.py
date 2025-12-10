import os
from time import sleep
import ray
import logging
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from common.cli import CommonTrainingArgs
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
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .env_runners(
            num_env_runners=args.env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
            num_cpus_per_learner=args.num_cpus_per_learner,
        )
        .multi_agent(
            **get_policy_config(args.mode)
        )
        .training(
            minibatch_size=args.minibatch_size,
        )
    )
    
    algo = config.build_algo()

    param_space = config.to_dict()

    param_space["lr"] = tune.grid_search(args.lr)
    param_space["gamma"] = tune.grid_search(args.gamma)
    param_space["entropy_coeff"] = tune.grid_search(args.entropy_coeff)
    param_space["train_batch_size"] = tune.grid_search(args.training_batch_size)
    param_space["num_epochs"] = tune.grid_search(args.epochs)

    tuner = tune.Tuner(
        config.algo_class,
        param_space=param_space,
        run_config=tune.RunConfig(
            storage_path=args.checkpoint_dir,
            stop={
                "training_iteration": args.iters
            },
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=args.save_interval,
                checkpoint_at_end=True,
            ),
        )

    )

    logging.info("Start training...")

    result_grid = tuner.fit()
    
    logging.info("Training completed.")

    ray.shutdown()
    