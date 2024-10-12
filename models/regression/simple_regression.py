from typing import Any, Dict, List

import torch
from torch import nn
from models.base_pytorch_model import BaseModel


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, batch):
        return self.linear(batch).squeeze()


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
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1)

    def forward(self, batch):
        batch = batch.unsqueeze(1)
        batch = batch.permute(0, 2, 1)
        batch = self.cnn(batch)
        batch = batch.view(batch.size(0), -1)  
        return batch
    
class BasicCNN(BaseModel):
    def __init__(self, input_dim):
        super().__init__(input_dim)

    def _build_network(self):
        return CNN(self.input_dim)
    
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}