from typing import Literal, List, Optional
from dataclasses import field
from argparse_dataclass import dataclass
from common.cli import CommonEnvArgs, CommonEvalArgs, CommonTrainingArgs


@dataclass
class EnvSpecificArgs(CommonEnvArgs):
    pass

@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    mode: Literal["shared_cnn", "cnn_vs_biased_random", "cnn_vs_mlp"]
    # cnn_hiddens: List[int] = field(metadata={"help": "Number of hidden units per layer", "nargs": "+"})

@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    pass