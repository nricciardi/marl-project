from typing import List
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from dsse_search.random.cli import EnvSpecificArgs, EvalArgs, TrainingArgs
from dsse_search.random.module.attention_module import DsseSearchAttentionRLModule
from dsse_search.random.module.cnn_mlp_fusion_module import DsseSearchCnnMlpFusionRLModule
from dsse_search.random.module.mlp_module import DsseSearchMlpRLModule
from dsse_search.random.module.mlp_module_v2 import DsseSearchMlpV2RLModule


def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:

    env_config = {
        "grid_size": args.grid_size,
        "timestep_limit": args.timestep_limit,
        "person_amount": args.person_amount,
        "dispersion_inc": args.dispersion_inc,
        "drone_amount": args.drone_amount,
        "drone_speed": args.drone_speed,
        "detection_probability": args.detection_probability
    }

    config = config.environment(
        env_name,
        env_config=env_config,
        clip_rewards=args.clip_rewards,
    )

    return config


def apply_policy_config(config: PPOConfig, mode: str) -> PPOConfig:
    
    if mode == "shared_cnn_mlp_fusion":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchCnnMlpFusionRLModule,
                            model_config={
                                "probability_matrix_cnn_conv2d": [1, 16, 32],
                                "probability_matrix_cnn_kernel_sizes": [3, 3, 3],
                                "probability_matrix_cnn_strides": [2, 2, 2],
                                "probability_matrix_cnn_paddings": [1, 1, 1],
                                "drone_coordinates_mlp_hiddens": [4, 4],
                                "drone_coordinates_mlp_dropout": 0,
                                "fusion_mlp_hiddens": [128, 64],
                                "fusion_mlp_dropout": 0,
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
    elif mode == "shared_mlp":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchMlpRLModule,
                            model_config={
                                "mlp_hiddens": [32, 64, 64, 64, 32],
                                "mlp_dropout": 0,
                            }
                        )   
                    )
                    .multi_agent(
                        policies={
                            "shared_mlp",
                        },
                        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_mlp"
                    )
                )
    
    elif mode == "shared_attention":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchAttentionRLModule,
                            model_config={
                            }
                        )   
                    )
                    .multi_agent(
                        policies={
                            "shared_mlp",
                        },
                        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_mlp"
                    )
                )
    
    elif mode == "shared_mlp_v2":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchMlpV2RLModule,
                            model_config={
                                "mlp_hiddens": [1024, 512, 256, 128, 64, 32, 16],
                                "mlp_dropout": 0,
                            }
                        )   
                    )
                    .multi_agent(
                        policies={
                            "shared_mlp_v2",
                        },
                        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_mlp_v2"
                    )
                )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_train_ppo_config(args: TrainingArgs, env_name: str) -> PPOConfig:
    config = initialize_base_training_ppo_from_args(args)
    config = apply_environment_config(config, args, env_name)
    config = apply_policy_config(
        config,
        mode=args.mode,
    )

    config = config.experimental(
        _disable_preprocessor_api=True
    )

    return config
