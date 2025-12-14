from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from dsse_search.cli import EnvSpecificArgs, EvalArgs, TrainingArgs
from dsse_search.module.cnn_module import DsseSearchCnnRLModule


def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:
    config = config.environment(
        env_name,
        env_config={
        },
        clip_rewards=args.clip_rewards,
    )

    return config


def apply_policy_config(config: PPOConfig, mode: str) -> PPOConfig:
    if mode == "shared_cnn":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchCnnRLModule,
                            model_config={
                                "cnn_layers": [32, 64, 128],
                                "mlp_layers": [256, 256],
                            }
                        )   
                    )
                    .multi_agent(
                        policies={
                            "shared_cnn",
                        },
                        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_cnn"
                    )
                )
    
    raise ValueError(f"Unknown mode: {mode}")


def get_train_ppo_config(args: TrainingArgs, env_name: str) -> PPOConfig:
    config = initialize_base_training_ppo_from_args(args)
    config = apply_environment_config(config, args, env_name)
    config = apply_policy_config(config, args.mode)

    return config
