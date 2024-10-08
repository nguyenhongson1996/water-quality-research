from typing import Any, Dict, List

import torch
from torch import nn
from models.base_pytorch_model import BaseModel


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        lstm_out, _ = self.lstm(batch)
        return self.fc(lstm_out).squeeze()


class BasicLSTM(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,  output_dim: int):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        super(BasicLSTM, self).__init__(input_dim)

    def _build_network(self) -> nn.Module:
        return LSTMNetwork(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        return {}
