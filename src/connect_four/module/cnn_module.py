from typing import Any, Dict, Optional

from common.network import build_cnn, build_mlp
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType


class Connect4CnnRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Custom PyTorch RLModule for Connect Four (New API Stack).
    Handles CNN processing and Action Masking.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        
        obs_space = self.observation_space["observation"]
        rows, cols, input_channels = obs_space.shape
        n_actions = cols  # One action per column

        cnn_conv2d = self.model_config.get("cnn_conv2d")
        cnn_strides = self.model_config.get("cnn_strides")
        cnn_kernel_sizes = self.model_config.get("cnn_kernel_sizes")
        cnn_paddings = self.model_config.get("cnn_paddings")

        self.cnn = nn.Sequential(
            *build_cnn(
                cnn_conv2d,
                cnn_strides,
                cnn_kernel_sizes,
                cnn_paddings,
                flat=True
            )
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, cnn_conv2d[0], rows, cols)
            cnn_output = self.cnn(dummy_input)
            cnn_output_dim = cnn_output.shape[1]

        mlp_hiddens = self.model_config.get("mlp_hiddens")
        mlp_dropout = self.model_config.get("mlp_dropout", 0.0)

        self.mlp = nn.Sequential(
            *build_mlp(
                mlp_hiddens=mlp_hiddens,
                input_dim=cnn_output_dim,
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
        
        obs_data = batch[Columns.OBS]
        
        observation = obs_data["observation"]
        
        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation)

        grid = observation.float()

        if grid.dim() == 3:
            grid = grid.unsqueeze(0)  # Add batch dimension if missing

        action_mask = obs_data["action_mask"]

        if not torch.is_tensor(action_mask):
            action_mask = torch.from_numpy(action_mask)

        # Permute for PyTorch CNN (Batch, H, W, C) -> (Batch, C, H, W)
        x = grid.permute(0, 3, 1, 2)

        # Forward Pass
        cnn_embeddings = self.cnn(x)
        embeddings = self.mlp(cnn_embeddings)
        logits = self.action_head(embeddings)

        action_mask = action_mask.to(logits.device)

        # Action Masking Logic
        masked_logits = logits.masked_fill(action_mask == 0, float("-inf"))

        return embeddings, masked_logits

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