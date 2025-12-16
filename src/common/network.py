from typing import List
import torch
import torch.nn as nn
import numpy as np


def build_cnn(cnn_conv2d: List[int], cnn_strides: List[int], 
                cnn_kernel_sizes: List[int], cnn_paddings: List[int],
                flat: bool = False) -> List[nn.Module]:
    
    assert len(cnn_conv2d) >= 2, "cnn_conv2d must specify at least input and one output channel."
    assert len(cnn_conv2d) == len(cnn_strides) == len(cnn_kernel_sizes) == len(cnn_paddings), \
        "cnn_strides, cnn_kernel_sizes, and cnn_paddings must have length equal to len(cnn_conv2d)."

    layers = []
    
    for i in range(1, len(cnn_conv2d)):
        current_conv2d = cnn_conv2d[i - 1]
        next_conv2d = cnn_conv2d[i]
        current_kernel_size = cnn_kernel_sizes[i - 1]
        current_stride = cnn_strides[i - 1]
        current_padding = cnn_paddings[i - 1]

        layers.append(nn.Conv2d(
            in_channels=current_conv2d, 
            out_channels=next_conv2d, 
            kernel_size=current_kernel_size,
            stride=current_stride,
            padding=current_padding,
        ))

        layers.append(nn.ReLU())

    if flat:
        layers.append(nn.Flatten())

    return layers


def build_mlp(mlp_hiddens: List[int], input_dim: int, dropout: float) -> List[nn.Module]:
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

    return layers