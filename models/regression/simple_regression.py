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
