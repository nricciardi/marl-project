from typing import Literal, List
from argparse_dataclass import dataclass
from common.cli import CommonEnvArgs, CommonEvalArgs, CommonTrainingArgs


@dataclass
class EnvSpecificArgs(CommonEnvArgs):
    n_walkers: int
    parallel_env: bool
    stacked_frames: int


@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]
    kl_coeff: List[int]
    fcnet_activation: str
    vf_share_layers: bool

@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]