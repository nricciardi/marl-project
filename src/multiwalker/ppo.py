from ray.rllib.algorithms.ppo import PPOConfig
from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from multiwalker.cli import EnvSpecificArgs, EvalArgs, TrainingArgs


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


def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:
    config = config.environment(
        env_name,
        env_config={
            "n_walkers": args.n_walkers,
            "parallel": args.parallel_env,
            "stacked_frames": args.stacked_frames,
        },
        clip_rewards=args.clip_rewards,
    )

    return config


def apply_policy_config(config: PPOConfig, mode: str) -> PPOConfig:
    config = config.multi_agent(
        **get_policy_config(mode)
    )
    return config


def get_train_ppo_config(args: TrainingArgs, env_name: str) -> PPOConfig:
    config = initialize_base_training_ppo_from_args(args)
    config = apply_environment_config(config, args, env_name)
    config = apply_policy_config(config, args.mode)

    return config
