from ray.rllib.algorithms.ppo import PPOConfig
from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from .cli import EnvSpecificArgs, EvalArgs, TrainingArgs
from ray import tune


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
            "n_agents": args.n_agents,
            "agent_radius": args.agent_radius,
            "agent_spacing": args.agent_spacing,
            "ball_radius": args.ball_radius,
            "wall_length": args.wall_length,
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

    model = {
        "fcnet_hiddens": args.fcnet_hiddens,
        "fcnet_activation": args.fcnet_activation,
        "vf_share_layers": args.vf_share_layers,
    }

    if args.kl_coeff is not None:
        model["kl_coeff"] = tune.grid_search(args.kl_coeff)

    config = config.training(
        model=model
    )

    return config
