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


class DsseSearchMlpRLModule(TorchRLModule, ValueFunctionAPI):
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

        mlp_hiddens = self.model_config.get("mlp_hiddens")
        mlp_dropout = self.model_config.get("mlp_dropout", 0.0)

        print("Drone Coordinates MLP Input Dim:", len(drone_coordinates))

        self.mlp = nn.Sequential(
            *build_mlp(
                mlp_hiddens=mlp_hiddens,
                input_dim=len(drone_coordinates) + 2 + 2,
                dropout=mlp_dropout
            )
        )

        output_dim = mlp_hiddens[-1]

        # Action Head (Policy) -> Logits
        self.action_head = nn.Linear(output_dim, n_actions)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

        # Value Head (Critic) -> Scalar
        self.value_head = nn.Linear(output_dim, 1)
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

        if probability_matrix.dim() == 2: # (Rows, Cols)
            probability_matrix = probability_matrix.unsqueeze(0)  # Add batch dimension

        probability_matrix = probability_matrix.float()     # (Batch, Rows, Cols)

        pdf = probability_matrix

        batch_size, height, width = probability_matrix.shape
        device = probability_matrix.device

        y_coords = torch.arange(height, device=device).float()
        x_coords = torch.arange(width, device=device).float()
        
        # indexing='ij' ensures grid_y varies along rows, grid_x along cols
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand to batch
        grid_y = grid_y.expand(batch_size, -1, -1)
        grid_x = grid_x.expand(batch_size, -1, -1)

        # E[x] = sum(x * p)
        com_y_float = (grid_y * pdf).sum(dim=(1, 2))
        com_x_float = (grid_x * pdf).sum(dim=(1, 2))

        # We round to the nearest pixel and convert to long (int64)
        com_y = torch.round(com_y_float).long().float()
        com_x = torch.round(com_x_float).long().float()

        # Note: Variance is calculated relative to the precise float center 
        # to minimize error, even if we return integer coordinates.
        # Reshape CoM for broadcasting: (Batch, 1, 1)
        com_y_expanded = com_y_float.view(batch_size, 1, 1)
        com_x_expanded = com_x_float.view(batch_size, 1, 1)

        var_y = (pdf * (grid_y - com_y_expanded) ** 2).sum(dim=(1, 2))
        var_x = (pdf * (grid_x - com_x_expanded) ** 2).sum(dim=(1, 2))



        if not torch.is_tensor(drone_coordinates):
            drone_coordinates = torch.from_numpy(drone_coordinates)
        
        if drone_coordinates.dim() == 1:  # (n_coordinates,)
            drone_coordinates = drone_coordinates.unsqueeze(0)  # (1, n_coordinates)

        drone_coordinates = drone_coordinates.float()  # (Batch, n_coordinates)

        # drone_coordinates = drone_coordinates / probability_matrix.shape[-1]  # Normalize coordinates to [0, 1]

        assert drone_coordinates.dim() == 2, "Drone coordinates tensor must be of shape (Batch, n_coordinates)."




        fusion_input = torch.cat([
            drone_coordinates,  # (Batch, n_coordinates)
            com_x.unsqueeze(1),
            com_y.unsqueeze(1),
            var_x.unsqueeze(1),
            var_y.unsqueeze(1),
        ], dim=-1)

        embeddings = self.mlp(fusion_input)

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