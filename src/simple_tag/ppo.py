from ray.rllib.algorithms.ppo import PPOConfig
from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from simple_tag.cli import EnvSpecificArgs, EvalArgs, TrainingArgs


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


def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:
    config = config.environment(
        env_name,
        env_config={
            "n_good_agents": args.n_good_agents,
            "n_bad_agents": args.n_bad_agents,
            "n_obstacles": args.n_obstacles,
            "max_cycles": args.max_cycles,
            "continuous_actions": args.continuous_actions,
        },
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


def get_eval_ppo_config(args: EvalArgs, env_name: str) -> PPOConfig:
    config = initialize_base_evaluation_ppo_from_args()
    config = apply_environment_config(config, args, env_name)
    config = apply_policy_config(config, args.mode)

    return config
