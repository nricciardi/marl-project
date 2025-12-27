from typing import Literal, List, Optional
from dataclasses import field
from argparse_dataclass import dataclass
from common.cli import CommonEnvArgs, CommonEvalArgs, CommonTrainingArgs
from dsse_search.common.cli import CommonDsseCnnTrainingArgs, CommonEnvSpecificArgs


@dataclass
class EnvSpecificArgs(CommonEnvSpecificArgs):
    pass
    

@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared_mlp", "shared_cnn_mlp_fusion", "shared_attention"]


@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    pass