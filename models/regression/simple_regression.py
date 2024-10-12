from typing import Any, Dict, List

import torch
from torch import nn
from models.base_pytorch_model import BaseModel


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, data):
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
            input_dim: number of features of input.
            out_channels: parameter, can be changed depended on 
            different purposes: combining a fully connected layer, etc. 
            kernel_size set to 1 because there exists a dimension of 
            input equal to 1.
        """
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
            Build CNN with PyTorch
            unsqueeze: Add a dimension to data so as to be suitable for 
            Conv1d (batch_size, input_dim) -> (batch_size, 1, input_dim).
            permute: change location of dimension so that input_dim turns 
            to be second dimension.
            view(data.size(0), -1): flatten to 2D.
        """
        data = data.unsqueeze(1)   
        data = data.permute(0, 2, 1)
        data = self.cnn(data)
        data = data.view(data.size(0), -1) 
        return data
    
class BasicCNN(BaseModel):
    def __init__(self, input_dim: int):
        super().__init__(input_dim)

    def _build_network(self):
        return CNN(self.input_dim)
    
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}
    