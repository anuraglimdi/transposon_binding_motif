"""
Module with PyTorch model classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class DNABindingCNN(nn.Module):
    """
    1-D Convolutional Neural Network for identifying DNA binding motif.
    """

    def __init__(
        self,
        model_params: dict,
    ):
        """
        All model parameters must be specified in the model_params dict
        """
        super(DNABindingCNN, self).__init__()
        self.model_params = model_params
        self.conv1 = nn.Conv1d(
            in_channels=self.model_params["alphabet_size"],
            out_channels=self.model_params["conv_channels"],
            kernel_size=self.model_params["conv_kernel_size"],
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.model_params["conv_channels"],
            out_channels=self.model_params["conv_channels"],
            kernel_size=self.model_params["conv_kernel_size"],
        )
        self.pool = nn.MaxPool1d(kernel_size=self.model_params["pool_kernel_size"])
        self.dropout = nn.Dropout(self.model_params["dropout_rate"])
        self.fc1 = nn.Linear(
            self._compute_flattened_size(),
            self.model_params["dense_layer_size"],
        )
        self.fc2 = nn.Linear(self.model_params["dense_layer_size"], 1)

    def _compute_flattened_size(self):
        # After conv1: L1 = sequence_length - kernel_size + 1
        L1 = (
            self.model_params["sequence_length"]
            - self.model_params["conv_kernel_size"]
            + 1
        )
        # After conv2: L2 = L1 - kernel_size + 1
        L2 = L1 - self.model_params["conv_kernel_size"] + 1
        # After pooling: L3 = L2 / pool size
        L3 = L2 // self.model_params["pool_kernel_size"]
        return self.model_params["conv_channels"] * L3  # number of filters after conv2

    def forward(self, x: torch.Tensor):
        # by default the one-hot encoded sequences are of shape
        # (length, alphabet size). For the convolutional layers,
        # want the shape to be (alphabet_size, length)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))  # Shape: (batch, conv_channels, L1)
        x = F.relu(self.conv2(x))  # Shape: (batch, conv_channels, L2)
        x = self.pool(x)  # Shape: (batch, conv_channels, L3)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
