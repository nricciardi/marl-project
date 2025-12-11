import os
import torch
import numpy as np
from typing import Any, Optional
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import logging
import imageio
from ray.tune.registry import register_env
from simple_tag.environment import environment_creator
from simple_tag.ppo import get_eval_ppo_config
from simple_tag.cli import EvalArgs
from argparse_dataclass import ArgumentParser


logging.basicConfig(level=logging.INFO)


def get_action_from_module(module, observation, is_continuous):
    obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
    
    with torch.no_grad():
        outputs = module.forward_inference({"obs": obs_tensor})
        dist_inputs = outputs["action_dist_inputs"][0] 

    if is_continuous:
        # Continuous: Take the mean (first half of the output)
        action_dim = dist_inputs.shape[0] // 2
        action = dist_inputs[:action_dim].numpy()
    else:
        # Discrete: Take the argmax
        action = torch.argmax(dist_inputs).item()
        
    return action


def visualize(
    algo: Algorithm, 
    env,
    n_episodes: int,
    *,
    video_dir: Optional[str] = None,
    fps: int = 15,
):
    """
    Evaluates the algorithm in the given environment and saves a video if requested.
    
    Args:
        algo: The trained RLlib algorithm (already built and restored).
        env: The PettingZoo environment instance (must be initialized with render_mode='rgb_array').
    """

    logging.info(f"Starting evaluation for {args.num_episodes} episodes...")

    if algo.config is None:
        raise ValueError("Algorithm configuration is not available.")

    policy_mapping_fn = algo.config.policy_mapping_fn

    if policy_mapping_fn is None:
        raise ValueError("Policy mapping function is not defined in the algorithm configuration.")

    # Ensure video directory exists if we need to save videos
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    observations, infos = env.reset()

    frames = []
    for episode_num in range(n_episodes):
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

        if video_dir:
            frame = env.render()
            frames.append(frame)

        if all(terminations.values()) or all(truncations.values()):
            observations, infos = env.reset()

    if video_dir and len(frames) > 0:
        video_path = os.path.join(video_dir, f"eval_ep_{episode_num + 1}.mp4")
        logging.info(f"Saving video to {video_path}...")
        
        try:
            imageio.mimsave(video_path, frames, fps=fps)
            logging.info("Video saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save video: {e}")

    


if __name__ == "__main__":
    args = ArgumentParser(EvalArgs).parse_args()

    logging.info("Initializing Ray...")
    ray.init()

    logging.info("Registering Simple Tag environment...")
    env_name = "simple_tag"
    register_env(env_name, lambda config: environment_creator(**config))
    
    config = get_eval_ppo_config(
        args=args,
        env_name=env_name,
    )

    algo = config.build_algo()

    logging.info(f"Restoring checkpoint from {args.checkpoint_path}...")
    algo.restore(args.checkpoint_path)

    env = environment_creator(
        n_good_agents=args.n_good_agents,
        n_bad_agents=args.n_bad_agents,
        n_obstacles=args.n_obstacles,
        max_cycles=args.max_cycles,
        continuous_actions=args.continuous_actions,
        render_mode="human"
    )

    visualize(
        algo,
        env,
        n_episodes=args.n_episodes,
        video_dir=args.video_dir,
    )

    





