from ray.rllib.algorithms.ppo import PPOConfig
from common.ppo import initialize_base_evaluation_ppo_from_args, initialize_base_training_ppo_from_args
from connect_four.cli import EnvSpecificArgs, EvalArgs, TrainingArgs
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from connect_four.module.cnn_module import Connect4CnnRLModule
from connect_four.module.biased_random_module import BiasedRandomConnect4RLModule
from connect_four.module.mlp_module import Connect4MlpRLModule


# class Connect4Callbacks(DefaultCallbacks):

#     def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        
#         # print("on_episode_end called")
#         # print(episode)
#         # print(dir(episode))
#         # print(episode.get_return())
#         # print(episode.get_rewards())

#         for player_id, rewards in episode.get_rewards().items():
#             for reward in rewards:
#                 metrics_logger.log_value(f"{player_id}_rewards", reward)




def apply_environment_config(config: PPOConfig, args: EnvSpecificArgs, env_name: str) -> PPOConfig:
    config = config.environment(
        env_name,
        env_config={
        },
        clip_rewards=args.clip_rewards,
    )

    return config


def apply_policy_config(config: PPOConfig, mode: str) -> PPOConfig:

    config = config

    if mode == "shared_cnn":
        return (config
                    .rl_module(
                        rl_module_spec=MultiRLModuleSpec(
                            rl_module_specs={
                                "custom_cnn": RLModuleSpec(
                                    module_class=Connect4CnnRLModule,
                                    model_config={
                                        "cnn_layers": [32, 64, 128],
                                    }
                                ),
                            }
                        )   
                    )
                    .multi_agent(
                        policies={
                            "custom_shared_cnn",
                        },
                        policy_mapping_fn=lambda agent_id, *args, **kwargs: "custom_shared_cnn"
                    )
                )
    
    elif mode == "cnn_vs_biased_random":
        return (config
                    .rl_module(
                        rl_module_spec=MultiRLModuleSpec(
                            rl_module_specs={
                                "custom_cnn": RLModuleSpec(
                                    module_class=Connect4CnnRLModule,
                                    model_config={
                                        "cnn_layers": [32, 64, 128],
                                    }
                                ),
                                "custom_biased_random": RLModuleSpec(
                                    module_class=BiasedRandomConnect4RLModule,
                                ),
                            }
                        )   
                    )
                    .multi_agent(
                        policies={
                            "custom_cnn",
                            "custom_biased_random",
                        },
                        policy_mapping_fn=lambda agent_id, *args, **kwargs: "custom_cnn" if agent_id == "player_0" else "custom_biased_random"
                    )
                )
    
    elif mode == "cnn_vs_mlp":

        return (config
                .rl_module(
                    rl_module_spec=MultiRLModuleSpec(
                        rl_module_specs={
                            "custom_cnn": RLModuleSpec(
                                module_class=Connect4CnnRLModule,
                                model_config={
                                    "cnn_layers": [32, 64, 128],
                                }
                            ),
                            "custom_mlp": RLModuleSpec(
                                module_class=Connect4MlpRLModule,
                                model_config={
                                    "fcnet_hiddens": [256, 256],
                                }
                            ),
                        }
                    )   
                )
                .multi_agent(
                    policies={
                        "custom_cnn",
                        "custom_mlp",
                    },
                    policy_mapping_fn=lambda agent_id, *args, **kwargs: "custom_cnn" if agent_id == "player_0" else "custom_mlp"
                )
            )


    else:
        raise ValueError(f"Unknown mode: {mode}")



def get_train_ppo_config(args: TrainingArgs, env_name: str) -> PPOConfig:
    config = initialize_base_training_ppo_from_args(args)
    config = apply_environment_config(config, args, env_name)
    config = apply_policy_config(config, args.mode)

    # config = config.callbacks(Connect4Callbacks)

    return config
