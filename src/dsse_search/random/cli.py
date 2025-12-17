from typing import Literal, List, Optional
from dataclasses import field
from argparse_dataclass import dataclass
from common.cli import CommonEnvArgs, CommonEvalArgs, CommonTrainingArgs


@dataclass
class EnvSpecificArgs(CommonEnvArgs):
    grid_size: int
    timestep_limit: int
    person_amount: int
    dispersion_inc: float
    drone_amount: int
    drone_speed: int
    detection_probability: float
    env_type: Literal["standard", "random_person_and_drone_initial_position"]
    


@dataclass
class TrainingArgs(CommonTrainingArgs, EnvSpecificArgs):
    mode: Literal["independent", "shared"]
    probability_matrix_cnn_conv2d: List[int] = field(metadata={"help": "Number of hidden units per layer", "nargs": "+"})
    probability_matrix_cnn_kernel_sizes: List[int] = field(metadata={"help": "Kernel sizes for each conv layer", "nargs": "+"})
    probability_matrix_cnn_strides: List[int] = field(metadata={"help": "Strides for each conv layer", "nargs": "+"})
    probability_matrix_cnn_paddings: List[int] = field(metadata={"help": "Paddings for each conv layer", "nargs": "+"})
    drone_coordinates_mlp_hiddens: List[int] = field(metadata={"help": "Number of hidden units per layer", "nargs": "+"})
    drone_coordinates_mlp_dropout: float = field(metadata={"help": "Dropout rate for the drone coordinates MLP"})
    fusion_mlp_hiddens: List[int] = field(metadata={"help": "Number of hidden units per layer", "nargs": "+"})
    fusion_mlp_dropout: float = field(metadata={"help": "Dropout rate for the fusion MLP"})


@dataclass
class EvalArgs(CommonEvalArgs, EnvSpecificArgs):
    pass