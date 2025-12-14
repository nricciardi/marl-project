from typing import Any, Dict, Optional
import math

import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class Connect4MlpRLModule(TorchRLModule, ValueFunctionAPI):
    """
    MLP RLModule for Connect Four.
    Flattens the grid and applies dense layers.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        obs_space = self.observation_space["observation"]
        
        # Calculate input size (Rows * Cols * Channels)
        input_dim = math.prod(obs_space.shape)

        hiddens = self.model_config.get("fcnet_hiddens")
        layers = []
        
        # Build MLP Tower
        prev_dim = input_dim
        for hidden_dim in hiddens:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.tower = nn.Sequential(*layers)

        # Action Head (Policy)
        self.action_head = nn.Linear(prev_dim, self.action_space.n)

        # Value Head (Critic)
        self.value_head = nn.Linear(prev_dim, 1)

    def _compute_embeddings_and_logits(self, batch: Dict[str, Any]):
        obs_data = batch[Columns.OBS]
        
        observation = obs_data["observation"]
        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation)

        grid = observation.float()

        # Add batch dimension if single observation (H, W, C) -> (1, H, W, C)
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)

        action_mask = obs_data["action_mask"]
        if not torch.is_tensor(action_mask):
            action_mask = torch.from_numpy(action_mask)

        # Flatten input: (Batch, H, W, C) -> (Batch, Input_Dim)
        x = grid.flatten(start_dim=1)

        # Forward Pass
        embeddings = self.tower(x)
        logits = self.action_head(embeddings)

        # Action Masking
        action_mask = action_mask.to(logits.device)
        masked_logits = logits.masked_fill(action_mask == 0, float("-inf"))

        return embeddings, masked_logits

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        _, logits = self._compute_embeddings_and_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        embeddings, logits = self._compute_embeddings_and_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_logits(batch)
            
        return self.value_head(embeddings).squeeze(-1)
