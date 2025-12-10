from dataclasses import field, dataclass
from typing import List, Literal, Union



@dataclass(kw_only=True)
class CommonTrainingArgs:
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
    
    # Training Loop settings
    iters: int = field(metadata={"help": "Stop after N iterations"})
    lr: List[float] = field(metadata={"help": "Learning rate(s) to sweep", "nargs": "+"})
    gamma: List[float] = field(metadata={"help": "Discount factor(s)", "nargs": "+"})
    training_batch_size: List[int] = field(metadata={"help": "Train batch size(s)", "nargs": "+"})
    epochs: List[int] = field(metadata={"help": "SGD epochs per iter", "nargs": "+"})
    entropy_coeff: List[float] = field(metadata={"help": "Entropy coefficient(s)", "nargs": "+"})
    minibatch_size: int = field(metadata={"help": "Minibatch size for SGD"})