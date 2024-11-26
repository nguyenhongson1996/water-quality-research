from typing import Any, Dict, List, Optional, Tuple

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

        self.model = self._build_network()

    def _build_network(self) -> nn.Module:
        """
        Builds the LSTM network.

        :return: The LSTM model.
        """
        return LSTMNetwork(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)

    def forward(self, batch: torch.Tensor, h0: Optional[torch.Tensor] = None, c0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        :param batch: Input batch.
        :param h0: Initial hidden state.
        :param c0: Initial cell state.
        :return: Model output.
        """
        if h0 is None:
            lstm_out, _ = self.model.lstm(batch)  # Use default h0 and c0
        else:
            lstm_out, _ = self.model.lstm(batch, (h0, c0))  # Use custom h0 and c0
        return self.model.fc(lstm_out).squeeze()  # Shape: (batch_size, seq_length, output_dim).

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

    def reconstruct_hidden_state(self, y: torch.Tensor) -> torch.Tensor:
        """
        Construct h0 from the given y using the inverse of the FC layer.
        :param y: Output value.
        :return: Reconstructed hidden state vector.
        """
        W = self.model.fc.weight.data  # Shape: (output_dim, hidden_dim).
        c = self.model.fc.bias.data    # Shape: (output_dim).

        batch_size = y.size(0)
        y = y.unsqueeze(1).expand(-1, W.size(0)) # Shape: (batch_size, output_dim).
        c = c.unsqueeze(0).expand(batch_size, -1) # Shape: (batch_size, output_dim).

        # Compute the pseudo-inverse of W
        W_inv = torch.pinverse(W)  # Shape: (hidden_dim, output_dim).
        # Calculate x from y: x = (y - c) * W^-1
        h0 = (y - c).mm(W_inv.t())  # Shape: (batch_size, hidden_dim).

        return h0.unsqueeze(0).repeat(self.num_layers, 1, 1) # Shape: (num_layers, batch_size, hidden_dim).

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

                y_prev = batch_y[:, 0]
                h0_new = self.reconstruct_hidden_state(y_prev)
                # Initialize c0 with the same size as h0
                c0_new = torch.zeros_like(h0_new) 
                # Pass the new h0 into the LSTM
                y_pred = self(batch_x, h0=h0_new, c0=c0_new)  

                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)  

            if val_loader:
                val_loss, detailed_report = self.evaluate(val_loader, loss_fn, detailed_eval_params, h0_new, c0_new)
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

    def evaluate(self, data_loader: DataLoader, loss_fn: nn.Module = nn.MSELoss(),
                 detailed_eval_params: Optional[Dict[str, Any]] = None, h0: Optional[torch.Tensor] = None, c0: Optional[torch.Tensor] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the model on the provided data loader.
        :param data_loader: Eval dataloader.
        :param loss_fn: Loss function.
        :param detailed_eval_params: Params used for calculating the detailed report.
        :return:
        """
        self.eval()
        detailed_eval_params = detailed_eval_params or {}
        total_loss = 0.0
        gts: List[torch.Tensor] = []
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                y_prev = batch_y[:, 0]
                h0 = self.reconstruct_hidden_state(y_prev)
                c0 = torch.zeros_like(h0) 
                y_pred = self(batch_x, h0, c0)
                loss = loss_fn(y_pred, batch_y)
                total_loss += loss.item()
                preds.append(y_pred)
                gts.append(batch_y)
        detailed_reports = self.calculate_detailed_report(preds, gts, **detailed_eval_params)
        return total_loss / len(data_loader), detailed_reports