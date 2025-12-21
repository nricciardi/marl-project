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


class DsseSearchMlpV2RLModule(TorchRLModule, ValueFunctionAPI):
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

        # (batch_size, n_drones + rows * cols, 3)

        self.mlp = nn.Sequential(
            *build_mlp(
                mlp_hiddens=mlp_hiddens,
                input_dim=len(drone_coordinates) + rows * cols,
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

        probability_matrix = probability_matrix.float()

        batch_size, rows, cols = probability_matrix.shape
        device = probability_matrix.device

        r_indices = torch.arange(rows, device=device).float() / rows
        c_indices = torch.arange(cols, device=device).float() / cols

        grid_r, grid_c = torch.meshgrid(r_indices, c_indices, indexing='ij')

        flat_r = grid_r.reshape(-1).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        flat_c = grid_c.reshape(-1).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        flat_probs = probability_matrix.reshape(batch_size, -1).unsqueeze(-1)
        result = torch.cat([flat_r, flat_c, flat_probs], dim=-1)

        if not torch.is_tensor(drone_coordinates):
            drone_coordinates = torch.from_numpy(drone_coordinates)

        if drone_coordinates.dim() == 1:
            drone_coordinates = drone_coordinates.unsqueeze(0)

        drone_coordinates = drone_coordinates.float()
        drone_coordinates = drone_coordinates.to(device)

        assert drone_coordinates.dim() == 2

        drones_xy = drone_coordinates.view(batch_size, -1, 2)
        drones_xy_norm = drones_xy.clone()
        drones_xy_norm[..., 0] = drones_xy_norm[..., 0] / rows
        drones_xy_norm[..., 1] = drones_xy_norm[..., 1] / cols

        drone_val = torch.ones((batch_size, drones_xy.shape[1], 1), device=device)

        drones_final = torch.cat([drones_xy_norm, drone_val], dim=-1)

        combined_input = torch.cat([result, drones_final], dim=1)   # (Batch, n_drones + rows * cols, 3)

        embeddings = self.mlp(combined_input)

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