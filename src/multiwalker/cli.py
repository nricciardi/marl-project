from typing import Literal
from argparse_dataclass import dataclass
from common.cli import CommonEvalArgs, CommonTrainingArgs


@dataclass
class EnvSpecificArgs:
    n_walkers: int


@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]

@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]