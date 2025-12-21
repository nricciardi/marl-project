from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

class DsseSearchAttentionRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Transformer-based RLModule using Dual Attention:
    1. Social Attention: Self-attention among drones to understand formation.
    2. Spatial Cross-Attention: Cross-attention between Ego-Drone and Grid Map.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        
        # --- Configuration ---
        n_actions = self.action_space.n
        
        # Hyperparameters (can be moved to model_config)
        self.d_model = 8
        self.n_heads = 2

        # --- Embeddings ---
        # Projects (x, y) coordinates to d_model
        self.drone_embedding = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.ReLU()
        )

        # Projects grid cell features (Probability, Relative_X, Relative_Y) to d_model
        self.grid_embedding = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.ReLU()
        )

        # --- Attention Modules ---
        # 1. Social Attention: Query = Ego Drone, Key/Value = All Drones
        self.social_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads, 
            batch_first=True
        )

        # 2. Spatial Attention: Query = Ego Drone, Key/Value = Grid Cells
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads, 
            batch_first=True
        )

        # --- Heads ---
        # Combines Social Context + Spatial Context
        fusion_dim = self.d_model * 2 
        
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Initialize weights for stability
        nn.init.orthogonal_(self.action_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)


    def _compute_embeddings_and_logits(self, batch: Dict[str, Any]):
        
        observation = batch[Columns.OBS]
        drone_coordinates_flat, probability_matrix = observation

        # --- Data Preparation ---
        
        # 1. Process Probability Matrix
        if not torch.is_tensor(probability_matrix):
            probability_matrix = torch.from_numpy(probability_matrix)
        if probability_matrix.dim() == 2:
            probability_matrix = probability_matrix.unsqueeze(0)
        
        probability_matrix = probability_matrix.float() # (Batch, Rows, Cols)
        batch_size, rows, cols = probability_matrix.shape
        grid_size = rows * cols

        # 2. Process Drone Coordinates
        if not torch.is_tensor(drone_coordinates_flat):
            drone_coordinates_flat = torch.from_numpy(drone_coordinates_flat)
        
        if drone_coordinates_flat.dim() == 1:
            drone_coordinates_flat = drone_coordinates_flat.unsqueeze(0)
            
        drone_coordinates_flat = drone_coordinates_flat.float()
        
        # Reshape flat coordinates to (Batch, N_Drones, 2)
        # We assume input is [x0, y0, x1, y1, ...]
        all_drones_pos = drone_coordinates_flat.view(batch_size, -1, 2)
        
        # Extract Ego Drone (Index 0)
        ego_drone_pos = all_drones_pos[:, 0:1, :] # (Batch, 1, 2)


        # --- Feature Engineering for Attention ---

        # A. Embed Drones (Social)
        # Query: Ego Drone, Key/Value: All Drones (Context)
        q_social = self.drone_embedding(ego_drone_pos)      # (Batch, 1, d_model)
        kv_social = self.drone_embedding(all_drones_pos)    # (Batch, N_Drones, d_model)

        # B. Embed Grid (Spatial)
        # Generate static grid coordinates (Batch, Rows, Cols, 2)
        # We create them on the fly to match device
        device = probability_matrix.device
        
        y_coords = torch.arange(rows, device=device).float()
        x_coords = torch.arange(cols, device=device).float()
        # meshgrid 'xy' indexing: x corresponds to cols, y to rows
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij') 
        grid_abs_coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) # (1, Rows, Cols, 2)
        
        # Flatten grid for attention: (Batch, N_Cells, ...)
        grid_abs_coords_flat = grid_abs_coords.view(1, -1, 2) # (1, 1600, 2)
        probs_flat = probability_matrix.view(batch_size, -1, 1) # (Batch, 1600, 1)

        # Calculate Relative Coordinates: Grid_Pos - Ego_Pos
        # Uses broadcasting: (1, 1600, 2) - (Batch, 1, 2) -> (Batch, 1600, 2)
        grid_relative_coords = grid_abs_coords_flat - ego_drone_pos

        # Normalize relative coords (Crucial for learning stability)
        grid_relative_coords = grid_relative_coords / grid_size

        # Concat: [Probability, Rel_X, Rel_Y]
        spatial_input = torch.cat([probs_flat, grid_relative_coords], dim=-1) # (Batch, 1600, 3)

        kv_spatial = self.grid_embedding(spatial_input) # (Batch, 1600, d_model)


        # --- Attention Passes ---

        # 1. Social Attention (Where are my teammates?)
        # We want the Ego drone to attend to other drones
        social_context, _ = self.social_attention(
            query=q_social, 
            key=kv_social, 
            value=kv_social
        ) # Output: (Batch, 1, d_model)

        # 2. Spatial Cross-Attention (Where should I look?)
        # Ego drone queries the grid map based on relative position and probability
        spatial_context, _ = self.spatial_attention(
            query=q_social, # Query is still the Ego Drone
            key=kv_spatial, 
            value=kv_spatial
        ) # Output: (Batch, 1, d_model)


        # --- Fusion & Output ---
        
        # Concatenate social and spatial knowledge
        fused_context = torch.cat([social_context, spatial_context], dim=-1) # (Batch, 1, d_model * 2)
        
        # Squeeze sequence dimension for the MLP heads
        fused_context = fused_context.squeeze(1) # (Batch, d_model * 2)

        logits = self.action_head(fused_context)
        embeddings = fused_context # Keep for potential loss usage

        return embeddings, logits


    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        _, logits = self._compute_embeddings_and_logits(batch)
        return {Columns.ACTION_DIST_INPUTS: logits}

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