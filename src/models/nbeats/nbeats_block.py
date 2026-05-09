"""N-BEATS block.
Will be used to build the N-BEATS model.
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn


class NBeatsBLock(nn.module):
    """
    An N-Beats block. It takes a lookback window flattened and outputs two vectors
    backcast
    forecast
    
    The block uses - A stack of fully-connected layers with ReLU activations.
      - Two linear heads that produce expansion coefficients (theta_b, theta_f).
      - Learnable basis matrices V (backcast) and W (forecast).
    """

    def __init__(
            self,
            input_size,
            output_size,
            hidden_layer_widths,
            theta_dim, #number of basis functions
    ):
        super().init()
        
        #stack
        layers = []
        in_features = input_size
        for units in hidden_layer_widths:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            in_features = units
        self.fc_stack = nn.Sequential(*layers)

        self.theta_back_head = nn.Linear(in_features, theta_dim)
        self.theta_fore_head = nn.Linear(in_features, theta_dim)

        self.backcast_basis = nn.Parameter(torch.randn(input_size, theta_dim))
        self.forecast_basis = nn.Parameter(torch.randn(output_size, theta_dim))

        nn.init.xavier_uniform_(self.backcast_basis)
        nn.init.xavier_uniform_(self.forecast_basis)

    def forward (self, x):

        """
        Args:
        x:  Tensor of shape (batch_size, input_size)
        Returns:
        backcast:  Tensor of shape (batch_size, input_size)
        forecast:  Tensor of shape (batch_size, output_size)
        """

        hidden = self.fc_stack(x)

        theta_b = self.theta_back_head(hidden)
        theta_f = self.theta_fore_head(hidden)

        backcast = torch.matmul(theta_b, self.backcast_basis.t())   # (batch, input_size)
        forecast = torch.matmul(theta_f, self.forecast_basis.t())   # (batch, output_size)
        
        return backcast, forecast









