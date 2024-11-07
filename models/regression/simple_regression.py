from typing import Any, Dict, List

import torch
from torch import nn
from models.base_pytorch_model import BaseModel


class LinearRegression(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.linear(data).squeeze()


class BasicRegression(BaseModel):
    def __init__(self, input_dim: int):
        super(BasicRegression, self).__init__(input_dim)

    def _build_network(self) -> nn.Module:
        return LinearRegression(self.input_dim)

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}

class CNN(nn.Module):
    def __init__(self, input_dim: int):
        """
        Basic CNN model.
        :param input_dim: Number of features of input.
        :param out_channels: Parameter that can be changed depending on different purposes, 
                             such as combining a fully connected layer, etc.
        :param kernel_size: Kernel size, arbitrary number, satisfy 2 * padding = kernel_size - 1. 
        :param padding: Padding.
        """
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=3, padding=1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
            Computing with CNN using PyTorch.    
        """
        data = data.unsqueeze(2)   
        # Add a dimension to data so as to be suitable for Conv1d (batch_size, input_dim) -> (batch_size, input_dim, 1).                                   
        data = data.permute(2,1,0) # Change (batch_size, input_dim, 1) -> (1, input_dim, batch_size) 
        data = self.cnn(data)
        data = data.view(data.size(0), -1) # flatten to 2D.
        return data
    
class BasicCNN(BaseModel):
    def __init__(self, input_dim: int):
        super().__init__(input_dim)

    def _build_network(self) -> nn.Module:
        return CNN(self.input_dim)
    
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}
    