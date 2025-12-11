import torch
from ray.rllib.algorithms.algorithm import Algorithm
import logging



def visualize(
    algo: Algorithm, 
    env,
    n_episodes: int,
):
    """
    Evaluates the algorithm in the given environment and saves a video if requested.
    
    Args:
        algo: The trained RLlib algorithm (already built and restored).
        env: The PettingZoo environment instance (must be initialized with render_mode='rgb_array').
    """

    logging.info(f"Starting evaluation for {n_episodes} episodes...")

    if algo.config is None:
        raise ValueError("Algorithm configuration is not available.")

    policy_mapping_fn = algo.config.policy_mapping_fn

    if policy_mapping_fn is None:
        raise ValueError("Policy mapping function is not defined in the algorithm configuration.")

    observations, infos = env.reset()

    for episode_num in range(n_episodes):
        while True:
            actions = {}
            for agent_id, agent_obs in observations.items():
                policy_id = policy_mapping_fn(agent_id)

                rl_module = algo.get_module(policy_id)

                fwd_ins = {"obs": torch.Tensor([agent_obs])}
                fwd_outputs = rl_module.forward_inference(fwd_ins)
                action_dist_class = rl_module.get_inference_action_dist_cls()
                action_dist = action_dist_class.from_logits(
                    fwd_outputs["action_dist_inputs"]
                )
                action = action_dist.sample()[0].numpy()
                actions[agent_id] = action

            logging.info(f"Actions taken: {actions}")

            observations, rewards, terminations, truncations, infos = env.step(actions)

            if all(terminations.values()) or all(truncations.values()):
                observations, infos = env.reset()
                break

    