from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType


class BiasedRandomConnect4RLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.action_dim = self.action_space.n

        self._param = nn.Parameter(torch.ones(self.action_dim) / self.action_dim, requires_grad=True)

    def _compute_random_logits(self, batch: Dict[str, Any]):
        obs_data = batch[Columns.OBS]
        
        action_mask = obs_data["action_mask"]
        
        if not torch.is_tensor(action_mask):
            action_mask = torch.from_numpy(action_mask)

        # Ensure the mask is on the same device as the model parameters
        action_mask = action_mask.to(self._param.device)

        # Expand Logits
        # self._param is assumed to be shape (A,)
        # - If action_mask is (A,), expand_as keeps it (A,) -> Dim 1 (Unbatched)
        # - If action_mask is (B, A), expand_as broadcasts it to (B, A) -> Dim 2 (Batched)
        logits = self._param.expand_as(action_mask)

        # Apply Masking
        # Set logits to -inf where action_mask is 0
        masked_logits = logits.masked_fill(action_mask == 0, float("-inf"))
        
        return masked_logits

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits = self._compute_random_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits = self._compute_random_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits = self._compute_random_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        batch_size = batch[Columns.OBS]["action_mask"].shape[0]
        device = batch[Columns.OBS]["action_mask"].device
        return torch.zeros(batch_size, device=device)