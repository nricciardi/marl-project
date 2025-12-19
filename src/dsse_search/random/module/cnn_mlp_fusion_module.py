from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from common.network import build_cnn, build_mlp


class DsseSearchCnnMlpFusionRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Custom PyTorch RLModule for Connect Four (New API Stack).
    Handles CNN processing and Action Masking.
    """


    @override(TorchRLModule)
    def setup(self) -> None:
        
        n_actions = self.action_space.n

        probability_matrix = self.observation_space[1]
        drone_coordinates = self.observation_space[0]

        rows, cols = probability_matrix.shape

        probability_matrix_cnn_conv2d = self.model_config.get("probability_matrix_cnn_conv2d")
        probability_matrix_cnn_strides = self.model_config.get("probability_matrix_cnn_strides")
        probability_matrix_cnn_kernel_sizes = self.model_config.get("probability_matrix_cnn_kernel_sizes")
        probability_matrix_cnn_paddings = self.model_config.get("probability_matrix_cnn_paddings")

        self.probability_matrix_cnn = nn.Sequential(
            *build_cnn(
                probability_matrix_cnn_conv2d,
                probability_matrix_cnn_strides,
                probability_matrix_cnn_kernel_sizes,
                probability_matrix_cnn_paddings,
                flat=True
            )
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, probability_matrix_cnn_conv2d[0], rows, cols)
            cnn_output = self.probability_matrix_cnn(dummy_input)
            cnn_output_dim = cnn_output.shape[1]

        drone_coordinates_mlp_hiddens = self.model_config.get("drone_coordinates_mlp_hiddens")
        drone_coordinates_mlp_dropout = self.model_config.get("drone_coordinates_mlp_dropout", 0.0)

        print("Drone Coordinates MLP Input Dim:", len(drone_coordinates))

        self.coordinates_mlp = nn.Sequential(
            *build_mlp(
                mlp_hiddens=drone_coordinates_mlp_hiddens,
                input_dim=len(drone_coordinates),   # (batch, n_coordinates)
                dropout=drone_coordinates_mlp_dropout
            )
        )

        fusion_mlp_hiddens = self.model_config.get("fusion_mlp_hiddens")
        fusion_mlp_dropout = self.model_config.get("fusion_mlp_dropout", 0.0)

        self.fusion_mlp = nn.Sequential(
            *build_mlp(
                mlp_hiddens=fusion_mlp_hiddens,
                input_dim=cnn_output_dim + drone_coordinates_mlp_hiddens[-1],
                dropout=fusion_mlp_dropout
            )
        )

        fusion_output_dim = fusion_mlp_hiddens[-1]


        # Action Head (Policy) -> Logits
        self.action_head = nn.Linear(fusion_output_dim, n_actions)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

        # Value Head (Critic) -> Scalar
        self.value_head = nn.Linear(fusion_output_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _compute_embeddings_and_logits(self, batch: Dict[str, Any]):
        """
        Helper function to process inputs and compute logits + mask.
        Used by both _forward and _forward_train.
        """
        
        observation = batch[Columns.OBS]

        drone_coordinates, probability_matrix = observation

        # # Debugging prints
        # print("Drone Coordinates:")
        # print(drone_coordinates)
        # print(drone_coordinates.shape)
        # print(len(drone_coordinates))
        # for coord in drone_coordinates:
        #     print(type(coord))
        #     print(coord.shape)
            # print(coord)
        # print("---")
        # print("Probability Matrix:")
        # print(probability_matrix)
        # print(probability_matrix.shape)
        # print("---")

        if not torch.is_tensor(probability_matrix):
            probability_matrix = torch.from_numpy(probability_matrix)

        if probability_matrix.dim() == 3: # (Batch, Rows, Cols)
            probability_matrix = probability_matrix.unsqueeze(1)  # Add channel dimension
        elif probability_matrix.dim() == 2: # (Rows, Cols)
            probability_matrix = probability_matrix.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        probability_matrix = probability_matrix.float()     # (Batch, 1, Rows, Cols)


        if not torch.is_tensor(drone_coordinates):
            drone_coordinates = torch.from_numpy(drone_coordinates)
        
        if drone_coordinates.dim() == 1:  # (n_coordinates,)
            drone_coordinates = drone_coordinates.unsqueeze(0)  # (1, n_coordinates)

        drone_coordinates = drone_coordinates.float()  # (Batch, n_coordinates)

        # drone_coordinates = drone_coordinates / probability_matrix.shape[-1]  # Normalize coordinates to [0, 1]

        assert drone_coordinates.dim() == 2, "Drone coordinates tensor must be of shape (Batch, n_coordinates)."

        cnn_output = self.probability_matrix_cnn(probability_matrix)
        coordinates_output = self.coordinates_mlp(drone_coordinates)
        fusion_input = torch.cat([cnn_output, coordinates_output], dim=-1)
        embeddings = self.fusion_mlp(fusion_input)

        logits = self.action_head(embeddings)

        return embeddings, logits

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Standard forward pass for inference/exploration.
        """
        _, logits = self._compute_embeddings_and_logits(batch)
        
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training. 
        We return embeddings here too, as some loss functions or architectures use them.
        """
        embeddings, logits = self._compute_embeddings_and_logits(batch)
        
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        """
        Computes the value function estimate.
        """
        # If embeddings are not passed (e.g. during inference), recompute them.
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_logits(batch)
            
        # Return value squeezed to shape (Batch,)
        return self.value_head(embeddings).squeeze(-1)