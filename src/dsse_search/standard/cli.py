from typing import Literal, List, Optional, Tuple
from dataclasses import field
from argparse_dataclass import dataclass
from common.cli import CommonEvalArgs, CommonTrainingArgs
from dsse_search.common.cli import CommonDsseCnnTrainingArgs, CommonEnvSpecificArgs


@dataclass
class EnvSpecificArgs(CommonEnvSpecificArgs):
    person_amount: int
    person_initial_position_x: int
    person_initial_position_y: int
    person_speed_x: float
    person_speed_y: float


@dataclass
class TrainingArgs(CommonDsseCnnTrainingArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]


@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    pass