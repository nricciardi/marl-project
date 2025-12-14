from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class EpisodeStats:
    episode_id: int
    total_reward: float
    length: int
    rewards_per_agent: Dict[str, float]


def simulate(
    algo: Algorithm, 
    env,
    n_episodes: int,
    explore: bool,
    sleep_time: float = 0.0
) -> List[EpisodeStats]:
    
    logging.info(f"Starting simulation for {n_episodes} episodes...")

    if algo.config is None:
        raise ValueError("Algorithm configuration is not available.")

    policy_mapping_fn = algo.config.policy_mapping_fn

    if policy_mapping_fn is None:
        raise ValueError("Policy mapping function is not defined in the algorithm configuration.")

    episode_stats_list = []

    for episode_num in range(n_episodes):
        observations, infos = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        agent_rewards = {agent_id: 0.0 for agent_id in env.agents}
        
        while True:
            if sleep_time > 0:
                time.sleep(sleep_time)

            actions = {}
            for agent_id, agent_obs in observations.items():
                policy_id = policy_mapping_fn(agent_id, episode_num) 
                
                rl_module = algo.get_module(policy_id)

                if rl_module is None:
                    raise ValueError(f"RL module for policy ID '{policy_id}' not found.")

                inputs = {
                    "obs": torch.Tensor([agent_obs]) if not isinstance(agent_obs, dict) else agent_obs,
                }
                
                with torch.no_grad():
                    if explore:
                        outputs = rl_module.forward_exploration(inputs)
                        action_dist = rl_module.get_exploration_action_dist_cls().from_logits(outputs[Columns.ACTION_DIST_INPUTS])
                    else:
                        outputs = rl_module.forward_inference(inputs)
                        action_dist = rl_module.get_inference_action_dist_cls().from_logits(outputs[Columns.ACTION_DIST_INPUTS])

                    action = action_dist.sample().squeeze(0).numpy()
                
                actions[agent_id] = action

                logging.info(f"Agent: {agent_id}, Policy: {policy_id}, Action: {action}")

            observations, rewards, terminations, truncations, infos = env.step(actions)
            logging.info(f"Step Rewards: {rewards}")
            
            # Aggregate rewards
            step_total_reward = sum(rewards.values())
            episode_reward += step_total_reward
            episode_length += 1
            
            for agent_id, reward in rewards.items():
                agent_rewards[agent_id] = agent_rewards.get(agent_id, 0) + reward

            if all(terminations.values()) or all(truncations.values()):
                if sleep_time > 0:
                    time.sleep(sleep_time)
                break
        
        stats = EpisodeStats(
            episode_id=episode_num,
            total_reward=episode_reward,
            length=episode_length,
            rewards_per_agent=agent_rewards
        )

        episode_stats_list.append(stats)

        logging.info(f"Episode {episode_num} finished. Total Reward: {episode_reward:.2f}")

    return episode_stats_list


def plot_simulation_results(stats: List[EpisodeStats], save_path: Optional[str] = None):
    rewards = [s.total_reward for s in stats]
    lengths = [s.length for s in stats]
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Setup the figure structure (2 subplots side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Reward per Episode (Trace)
    ax1.plot(rewards, marker='o', linestyle='-', color='royalblue', alpha=0.8, label='Episode Reward')
    ax1.axhline(mean_reward, color='crimson', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.1f}')
    ax1.fill_between(range(len(rewards)), mean_reward - std_reward, mean_reward + std_reward, color='crimson', alpha=0.1)
    
    ax1.set_title("Reward Trace (Stability Check)")
    ax1.set_xlabel("Episode ID")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Reward Distribution (Histogram)
    # This helps identify if you have the "floor" problem (many episodes at -300)
    sns.histplot(rewards, kde=True, ax=ax2, color='darkorange', bins=15)
    ax2.axvline(mean_reward, color='crimson', linestyle='--', linewidth=2)
    
    ax2.set_title("Reward Distribution (Mode Check)")
    ax2.set_xlabel("Total Reward")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Evaluation Results over {len(stats)} Episodes", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()