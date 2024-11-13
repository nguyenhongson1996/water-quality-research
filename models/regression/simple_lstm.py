from typing import Any, Dict, List, Optional

import torch
from torch import nn
from models.base_pytorch_model import BaseModel
from torch.utils.data.dataloader import DataLoader


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        """
        Basic LSTM network.
        :param input_dim: Input dimension.
        :param hidden_dim: Hidden state dimension.
        :param num_layers: Number of LSTM layers.
        :param output_dim: Output dimension.
        """
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  # LSTM layer.
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer.

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param batch: Input batch.
        :return: Model output.
        """
        lstm_out, _ = self.lstm(batch)  # Input shape: (batch_size, seq_length, hidden_dim).
        return self.fc(lstm_out).squeeze()  # Shape: (batch_size, seq_length, output_dim).


class BasicLSTM(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int,
                 seq_length: int):
        """
        Basic LSTM model.

        :param input_dim: Input dimension.
        :param hidden_dim: Hidden state dimension.
        :param num_layers: Number of LSTM layers.
        :param output_dim: Output dimension.
        :param seq_length: Sequence length.
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_length = seq_length  # Sequence length for data preparation.
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

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int = 100,
            optimizer_type: str = "adam", loss_fn: nn.Module = nn.MSELoss(), lr: float = 0.001,
            scheduler_type: str = "linear", scheduler_params: Optional[Dict[str, Any]] = None,
            patience: int = 10, detailed_eval_params: Optional[Dict[str, Any]] = None):
        """
        Train the model using the provided data loaders.

        :param train_loader: Train dataloader.
        :param val_loader: If provided, the model will be evaluated using this dataloader.
        :param epochs: Number of training epochs.
        :param optimizer_type: Optimizer type.
        :param loss_fn: Loss function.
        :param lr: Initial learning rate.
        :param scheduler_type: Learning rate scheduler type to use.
        :param scheduler_params: Learning rate scheduler parameters.
        :param patience: Number of patience epochs before early stopping.
        :param detailed_eval_params: Params used in evaluating the result in detail.
        """
        optimizer = self._get_optimizer(optimizer_type, lr)
        scheduler = self._get_lr_scheduler(optimizer, scheduler_type, scheduler_params)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = self(batch_x)  # Predict the output for the entire sequence.
                # Calculate loss for each element in the sequence.
                loss = loss_fn(y_pred.view(-1, self.output_dim), batch_y.view(-1, self.output_dim))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)  # Compute the average loss for the entire train set.

            if val_loader:
                val_loss, detailed_report = self.evaluate(val_loader, loss_fn, detailed_eval_params)
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if detailed_report:
                    print("Detailed report:")
                    for key, val in detailed_report.items():
                        print(f"{key}: {val}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    break

                # Step the scheduler.
                if scheduler_type == 'reduce_lr_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
                if scheduler_type != 'reduce_lr_on_plateau':
                    scheduler.step()

            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
