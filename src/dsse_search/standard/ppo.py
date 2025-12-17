from typing import List
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from dsse_search.standard.cli import EnvSpecificArgs, EvalArgs, TrainingArgs
from dsse_search.standard.module.cnn_mlp_fusion_module import DsseSearchCnnMlpFusionRLModule


def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:

    env_config = {
        "grid_size": args.grid_size,
        "timestep_limit": args.timestep_limit,
        "person_amount": args.person_amount,
        "dispersion_inc": args.dispersion_inc,
        "drone_amount": args.drone_amount,
        "drone_speed": args.drone_speed,
        "detection_probability": args.detection_probability,
    }

    config = config.environment(
        env_name,
        env_config=env_config,
        clip_rewards=args.clip_rewards,
    )

    return config


def apply_policy_config(config: PPOConfig, mode: str, probability_matrix_cnn_conv2d: List[int], probability_matrix_cnn_kernel_sizes: List[int],
                        probability_matrix_cnn_strides: List[int], probability_matrix_cnn_paddings: List[int],
                        drone_coordinates_mlp_hiddens: List[int], drone_coordinates_mlp_dropout: float, fusion_mlp_hiddens: List[int],
                        fusion_mlp_dropout: float) -> PPOConfig:
    if mode == "shared":
        return (config
                    .rl_module(
                        rl_module_spec=RLModuleSpec(
                            module_class=DsseSearchCnnMlpFusionRLModule,
                            model_config={
                                "probability_matrix_cnn_conv2d": probability_matrix_cnn_conv2d,
                                "probability_matrix_cnn_kernel_sizes": probability_matrix_cnn_kernel_sizes,
                                "probability_matrix_cnn_strides": probability_matrix_cnn_strides,
                                "probability_matrix_cnn_paddings": probability_matrix_cnn_paddings,
                                "drone_coordinates_mlp_hiddens": drone_coordinates_mlp_hiddens,
                                "drone_coordinates_mlp_dropout": drone_coordinates_mlp_dropout,
                                "fusion_mlp_hiddens": fusion_mlp_hiddens,
                                "fusion_mlp_dropout": fusion_mlp_dropout,
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
    config = apply_policy_config(
        config,
        mode=args.mode,
        probability_matrix_cnn_conv2d=args.probability_matrix_cnn_conv2d,
        probability_matrix_cnn_kernel_sizes=args.probability_matrix_cnn_kernel_sizes,
        probability_matrix_cnn_strides=args.probability_matrix_cnn_strides,
        probability_matrix_cnn_paddings=args.probability_matrix_cnn_paddings,
        drone_coordinates_mlp_hiddens=args.drone_coordinates_mlp_hiddens,
        drone_coordinates_mlp_dropout=args.drone_coordinates_mlp_dropout,
        fusion_mlp_hiddens=args.fusion_mlp_hiddens,
        fusion_mlp_dropout=args.fusion_mlp_dropout
    )

    return config
