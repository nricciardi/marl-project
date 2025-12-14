from dataclasses import field, dataclass
from typing import List, Literal, Optional, Union


@dataclass(kw_only=True)
class CommonEnvArgs:
    clip_rewards: bool = field(default=True, metadata={"help": "Whether to clip rewards"})


@dataclass(kw_only=True)
class CommonArgs:
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed for training"})


@dataclass(kw_only=True)
class CommonTrainingArgs(CommonArgs):
    """
    Base arguments for training.
    Hyperparameters are Lists to enable Ray Tune Grid Search.
    If a single value is desired, simply pass one value.
    """

    # System settings
    checkpoint_dir: str = field(metadata={"help": "Directory to store results"})
    env_runners: int = field(metadata={"help": "Number of rollout workers"})
    num_envs_per_env_runner: int = field(metadata={"help": "Envs per worker"})
    num_cpus_per_env_runner: int = field(metadata={"help": "Number of CPUs per env runner"})
    num_gpus_per_env_runner: float = field(metadata={"help": "Number of GPUs per env runner"})
    num_learners: int = field(metadata={"help": "Number of learner workers"})
    num_gpus_per_learner: float = field(metadata={"help": "Number of GPUs per learner worker"})
    num_cpus_per_learner: int = field(metadata={"help": "Number of CPUs per learner worker"})
    save_interval: int = field(metadata={"help": "Checkpoint frequency"})
    
    from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Directory to store results"})


    # Training Loop settings
    iters: int = field(metadata={"help": "Stop after N iterations"})
    lr: List[float] = field(metadata={"help": "Learning rate(s) to sweep", "nargs": "+"})
    gamma: List[float] = field(metadata={"help": "Discount factor(s)", "nargs": "+"})
    training_batch_size: List[int] = field(metadata={"help": "Train batch size(s)", "nargs": "+"})
    epochs: List[int] = field(metadata={"help": "SGD epochs per iter", "nargs": "+"})
    entropy_coeff: List[float] = field(metadata={"help": "Entropy coefficient(s)", "nargs": "+"})
    minibatch_size: List[int] = field(metadata={"help": "Minibatch size(s) for SGD", "nargs": "+"})
    observation_filter: Literal["NoFilter", "MeanStdFilter"] = field(default="NoFilter", metadata={"help": "Observation filter to use"})
    lambda_: List[float] = field(metadata={"args": ["--lambda"], "help": "Lambda", "nargs": "+"})
    clip_param: List[float] = field(metadata={"help": "Clip param(s)", "nargs": "+"})

    # Evaluation settings
    evaluation_interval: int = field(default=0, metadata={"help": "Evaluate every N iterations. 0 to disable."})
    evaluation_duration: int = field(default=10, metadata={"help": "Number of episodes for each evaluation."})
    evaluation_duration_unit: Literal["episodes", "timesteps"] = field(default="episodes", metadata={"help": "Unit for evaluation duration."})



@dataclass(kw_only=True)
class CommonEvalArgs(CommonArgs):
    """
    Base arguments for evaluation.
    """
    checkpoint_path: str = field(metadata={"help": "Path to the checkpoint to evaluate"})
    n_episodes: int = field(default=10, metadata={"help": "Number of episodes to evaluate"})
    explore: bool = field(default=False, metadata={"help": "Explore during evaluation"})
    sleep_time: float = field(default=0.0, metadata={"help": "Sleep time between steps (in seconds)"})