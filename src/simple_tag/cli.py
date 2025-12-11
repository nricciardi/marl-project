from typing import Literal
from argparse_dataclass import dataclass
from common.cli import CommonEvalArgs, CommonTrainingArgs


@dataclass
class EnvSpecificArgs:
    n_good_agents: int
    n_bad_agents: int
    n_obstacles: int
    max_cycles: int
    continuous_actions: bool


@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    # "group_shared" = Bad agents share a policy, Good agents share a different policy
    mode: Literal["independent", "group_shared"]

@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    mode: Literal["independent", "group_shared"]