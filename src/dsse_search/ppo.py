from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from dsse_search.cli import EnvSpecificArgs, EvalArgs, TrainingArgs
from dsse_search.module.cnn_mlp_fusion_module import DsseSearchCnnMlpFusionRLModule


def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:
    config = config.environment(
        env_name,
        env_config={
        },
        clip_rewards=args.clip_rewards,
    )

    return config


def apply_policy_config(config: PPOConfig, mode: str) -> PPOConfig:
    if mode == "shared":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchCnnMlpFusionRLModule,
                            model_config={
                                "probability_matrix_cnn_conv2d": [
                                    # in_channels, kernel_size, stride, padding
                                    [1, 3, 1, 1],
                                    [16, 3, 1, 1],
                                    [32, 3, 1, 1],
                                    [64, 3, 1, 1],
                                ],
                                "drone_coordinates_mlp_hiddens": [16, 32],
                                "drone_coordinates_mlp_dropout": 0,
                                "fusion_mlp_hiddens": [128, 64],
                                "fusion_mlp_dropout": 0.2,
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
