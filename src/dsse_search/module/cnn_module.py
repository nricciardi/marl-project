from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType


class DsseSearchCnnRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Custom PyTorch RLModule for Connect Four (New API Stack).
    Handles CNN processing and Action Masking.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        
        obs_space = self.observation_space["observation"]
        rows, cols, input_channels = obs_space.shape

        cnn_hiddens = [input_channels] + self.model_config.get("cnn_layers")
        mlp_hiddens = self.model_config.get("mlp_layers")

        layers = []
        
        for i in range(1, len(cnn_hiddens)):
            in_c = cnn_hiddens[i - 1]
            out_c = cnn_hiddens[i]
            
            layers.append(nn.Conv2d(
                in_channels=in_c, 
                out_channels=out_c, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ))
            layers.append(nn.ReLU())

            # TODO: Consider adding pooling layers or residual block if needed
        
        layers.append(nn.Flatten())

        flattened_size = rows * cols * cnn_hiddens[-1]

        layers.append(nn.LayerNorm(flattened_size))

        prev_dim = flattened_size
        for hidden_dim in mlp_hiddens:
            layer = nn.Linear(prev_dim, hidden_dim)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
            
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.tower = nn.Sequential(*layers)

        # Action Head (Policy) -> Logits
        self.action_head = nn.Linear(mlp_hiddens[-1], self.action_space.n)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

        # Value Head (Critic) -> Scalar
        self.value_head = nn.Linear(mlp_hiddens[-1], 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _compute_embeddings_and_logits(self, batch: Dict[str, Any]):
        """
        Helper function to process inputs and compute logits + mask.
        Used by both _forward and _forward_train.
        """

        print("Batch contents:")
        print(batch)
        
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
        embeddings = self.tower(x)
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