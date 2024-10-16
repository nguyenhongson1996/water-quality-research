from typing import Any, Dict, List, Optional

import torch
from torch import nn
from models.base_pytorch_model import BaseModel


class LSTMNetwork(nn.Module):
    """
    Basic LSTM network.

    :param input_dim: Input dimension.
    :param hidden_dim: Hidden state dimension.
    :param num_layers: Number of LSTM layers.
    :param output_dim: Output dimension.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param batch: Input batch.
        :return: Model output.
        """
        lstm_out, _ = self.lstm(batch)  # Run LSTM
        return self.fc(lstm_out[:, -1, :]).squeeze()  # Return the last output of the sequence


class BasicLSTM(BaseModel):
    """
    Basic LSTM model.

    :param input_dim: Input dimension.
    :param hidden_dim: Hidden state dimension.
    :param num_layers: Number of LSTM layers.
    :param output_dim: Output dimension.
    :param seq_length: Sequence length.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int,
                 seq_length: int):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_length = seq_length  # Sequence length for data preparation
        super(BasicLSTM, self).__init__(input_dim)

    def _build_network(self) -> nn.Module:
        """
        Builds the LSTM network.

        :return: The LSTM model.
        """
        return LSTMNetwork(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)

    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        """
        Calculate detailed evaluation report (Not implemented).
        :param predictions: List of prediction tensors.
        :param ground_truth: List of ground truth tensors.
        :param kwargs: Additional keyword arguments.
        :return: Detailed evaluation report.
        """
        return {}