from typing import Any, Dict, Optional

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
        """
        Builds the model components. Replaces __init__ in the old API.
        """
        
        obs_space = self.observation_space["observation"]
        rows, cols, input_channels = obs_space.shape

        hiddens = [input_channels] + self.model_config.get("cnn_layers")
        layers = []
        
        for i in range(1, len(hiddens)):
            in_c = hiddens[i - 1]
            out_c = hiddens[i]
            
            layers.append(nn.Conv2d(
                in_channels=in_c, 
                out_channels=out_c, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ))
            layers.append(nn.ReLU())
        
        layers.append(nn.Flatten())
        self.tower = nn.Sequential(*layers)

        # Calculate Flattened Size
        self.flattened_size = rows * cols * hiddens[-1]

        # Action Head (Policy) -> Logits
        self.action_head = nn.Linear(self.flattened_size, self.action_space.n)

        # Value Head (Critic) -> Scalar
        self.value_head = nn.Linear(self.flattened_size, 1)

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