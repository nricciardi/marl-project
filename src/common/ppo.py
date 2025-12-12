from ray.rllib.algorithms.ppo import PPOConfig

from common.cli import CommonTrainingArgs


def initialize_base_training_ppo_from_args(args: CommonTrainingArgs) -> PPOConfig:
    config = (
        PPOConfig()
            .framework("torch")
            .env_runners(
                num_env_runners=args.env_runners,
                num_envs_per_env_runner=args.num_envs_per_env_runner,
                num_cpus_per_env_runner=args.num_cpus_per_env_runner,
                num_gpus_per_env_runner=args.num_gpus_per_env_runner,
                observation_filter=args.observation_filter
            )
            .learners(
                num_learners=args.num_learners,
                num_gpus_per_learner=args.num_gpus_per_learner,
                num_cpus_per_learner=args.num_cpus_per_learner,
            )
        )
    
    return config


def initialize_base_evaluation_ppo_from_args() -> PPOConfig:
    config = (
        PPOConfig()
            .framework("torch")
            .env_runners(
                num_env_runners=0,
                num_envs_per_env_runner=1,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0.3,
            )
        )
    
    return config