import ray
import logging
from evaluate.visualize import visualize
from evaluate.simulate import simulate, plot_simulation_results
from ray.tune.registry import register_env
from .environment import environment_creator
from .cli import EvalArgs
from argparse_dataclass import ArgumentParser
from ray.rllib.algorithms.algorithm import Algorithm


logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    args = ArgumentParser(EvalArgs).parse_args()

    logging.info("Initializing Ray...")
    ray.init()

    env_name = "vmas_buzz_wire"
    register_env(env_name, lambda config: environment_creator(**config))
    
    logging.info(f"Restoring checkpoint from {args.checkpoint_path}...")
    algo = Algorithm.from_checkpoint(args.checkpoint_path)

    env = environment_creator(
        n_agents=args.n_agents,
        agent_radius=args.agent_radius,
        agent_spacing=args.agent_spacing,
        ball_radius=args.ball_radius,
        wall_length=args.wall_length,
        stacked_frames=args.stacked_frames,
        continuous_actions=args.continuous_actions,
        render_mode="human",
    )

    results = simulate(
        algo,
        env,
        n_episodes=args.n_episodes,
        explore=args.explore,
    )

    plot_simulation_results(results)

    





