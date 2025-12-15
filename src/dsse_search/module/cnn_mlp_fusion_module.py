from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class DsseSearchCnnMlpFusionRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Custom PyTorch RLModule for Connect Four (New API Stack).
    Handles CNN processing and Action Masking.
    """

    def __build_cnn(self, cnn_conv2d: List[List[int]]) -> nn.Sequential:
        layers = []
        
        for i in range(1, len(cnn_conv2d)):
            current_conv2d = cnn_conv2d[i - 1]
            next_conv2d = cnn_conv2d[i]
            
            layers.append(nn.Conv2d(
                in_channels=current_conv2d[0], 
                out_channels=next_conv2d[0], 
                kernel_size=current_conv2d[1],
                stride=current_conv2d[2],
                padding=current_conv2d[3],
            ))

            layers.append(nn.ReLU())

            # TODO: Consider adding pooling layers or residual block if needed
        
        layers.append(nn.Flatten())

        return nn.Sequential(*layers)
    
    def __build_mlp(self, mlp_hiddens: List[int], input_dim: int, dropout: float) -> nn.Sequential:
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in mlp_hiddens:
            layer = nn.Linear(prev_dim, hidden_dim)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
            
            layers.append(layer)
            layers.append(nn.ReLU())
            
            prev_dim = hidden_dim
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)


    @override(TorchRLModule)
    def setup(self) -> None:
        
        n_actions = self.action_space.n

        drone_coordinates, probability_matrix = self.observation_space

        rows, cols = probability_matrix.shape
        n_coordinates = drone_coordinates.shape[0]

        probability_matrix_cnn_conv2d = self.model_config.get("probability_matrix_cnn_conv2d")

        self.probability_matrix_cnn = self.__build_cnn(probability_matrix_cnn_conv2d)

        flattened_size = rows * cols * probability_matrix_cnn_conv2d[-1][0]

        drone_coordinates_mlp_hiddens = self.model_config.get("drone_coordinates_mlp_hiddens")
        drone_coordinates_mlp_dropout = self.model_config.get("drone_coordinates_mlp_dropout", 0.0)

        self.coordinates_mlp = self.__build_mlp(
            drone_coordinates_mlp_hiddens,
            n_coordinates,
            drone_coordinates_mlp_dropout
        )

        fusion_mlp_hiddens = self.model_config.get("fusion_mlp_hiddens")
        fusion_mlp_dropout = self.model_config.get("fusion_mlp_dropout", 0.0)

        self.fusion_mlp = self.__build_mlp(
            fusion_mlp_hiddens,
            flattened_size + drone_coordinates_mlp_hiddens[-1],
            fusion_mlp_dropout
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

        print("Drone Coordinates:")
        print(drone_coordinates)
        for coord in drone_coordinates:
            print(coord)
            print(coord.shape)
        print("---")
        print("Probability Matrix:")
        print(probability_matrix)
        print(probability_matrix.shape)
        print("---")

        if not torch.is_tensor(drone_coordinates):
            drone_coordinates = torch.tensor(drone_coordinates)

        while drone_coordinates.dim() < 2: # Add batch dimension if missing
            drone_coordinates = drone_coordinates.unsqueeze(0)

        if not torch.is_tensor(probability_matrix):
            probability_matrix = torch.from_numpy(probability_matrix)

        probability_matrix = probability_matrix.float()

        while probability_matrix.dim() < 4: # Add batch dimension if missing
            probability_matrix = probability_matrix.unsqueeze(0) 

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