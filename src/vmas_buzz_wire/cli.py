from typing import Literal, List, Optional
from dataclasses import field
from argparse_dataclass import dataclass
from common.cli import CommonEnvArgs, CommonEvalArgs, CommonTrainingArgs


@dataclass
class EnvSpecificArgs(CommonEnvArgs):
    n_agents: int
    wall_length: float
    agent_radius: float
    agent_spacing: float
    ball_radius: float
    stacked_frames: int
    continuous_actions: bool

@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]
    fcnet_hiddens: List[int] = field(metadata={"help": "Number of hidden units per layer", "nargs": "+"})
    fcnet_activation: str
    vf_share_layers: bool
    kl_coeff: List[float] = field(default_factory=list, metadata={"help": "KL coefficient(s)", "nargs": "+"})

@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]