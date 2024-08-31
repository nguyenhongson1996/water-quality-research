from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau, StepLR
from torch.utils.data.dataloader import DataLoader


class BaseModel(nn.Module):
    def __init__(self, input_dim: int):
        """
        Basic regression model that implements basic functions.
        :param input_dim: Input dimension.
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.network = self._build_network()

    @abstractmethod
    def _build_network(self) -> nn.Module:
        """
        Construct the network architecture.
        """
        raise NotImplementedError()

    def _get_optimizer(self, optimizer_type: str, lr: float) -> torch.optim.Optimizer:
        """
        Get model optimizer.
        :param optimizer_type: Optimizer type.
        :param lr: Initial learning rate.
        :return: Optimizer.
        """
        if optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == "adamW":
            return torch.optim.AdamW(self.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer type {optimizer_type} is not supported.")

    def _get_lr_scheduler(self, optimizer: torch.optim.Optimizer, scheduler_type: str,
                          scheduler_params: Optional[Dict[str, Any]]) -> LRScheduler:
        """
        Get learning rate scheduler.
        :param optimizer: Optimizer.
        :param scheduler_type: Learning rate scheduler type.
        :param scheduler_params: Parameters.
        :return: Learning rate scheduler instance.
        """
        if scheduler_params is None:
            scheduler_params = {}

        if scheduler_type == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(optimizer, **scheduler_params)
        elif scheduler_type == 'step_lr':
            return StepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'cosine_annealing':
            return CosineAnnealingLR(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Process an input batch.
        :param batch: Input batch.
        :return: Regression values.
        """
        return self.network(batch)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int = 100,
            optimizer_type: str = "adam", loss_fn: nn.Module = nn.MSELoss(), lr: float = 0.001,
            scheduler_type: str = "step_lr", scheduler_params: Optional[Dict[str, Any]] = None,
            patience: int = 10, detailed_eval_params: Optional[Dict[str, Any]] = None):
        """
        Train the model using the provided data loaders.

        :param train_loader: Train dataloader.
        :param val_loader: If provided, the model will be evaluated using this dataloader.
        :param epochs: Number of training epoch.
        :param optimizer_type: Optimizer type.
        :param loss_fn: Loss function.
        :param lr: Initial learning rate.
        :param scheduler_type: Learning rate scheduler type to use.
        :param scheduler_params: Learning rate scheduler parameters.
        :param patience: Number of patience epoch before early stopping.
        :param detailed_eval_params: Params used in evaluating the result in details.
        """
        optimizer = self._get_optimizer(optimizer_type, lr)
        scheduler = self._get_scheduler(optimizer, scheduler_type, scheduler_params)

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

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

    @abstractmethod
    def calculate_detailed_report(self, predictions: List[torch.Tensor], ground_truth: List[torch.Tensor],
                                  **kwargs) -> Dict[str, Any]:
        """
        Calculate the detailed report.
        """
        raise NotImplementedError()

    def evaluate(self, data_loader: DataLoader, loss_fn: nn.Module,
                 detailed_eval_params: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the model on the provided data loader.
        :param data_loader: Eval dataloader.
        :param loss_fn: Loss function.
        :param detailed_eval_params: Params used for calculating the detailed report.
        :return:
        """
        self.eval()
        total_loss = 0.0
        gts: List[torch.Tensor] = []
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                y_pred = self(batch_x)
                loss = self.calculate_loss(y_pred, batch_y, loss_fn)
                total_loss += loss.item()
                preds.append(y_pred)
                gts.append(batch_y)
        detailed_reports = self.calculate_detailed_report(preds, gts, **detailed_eval_params)
        return total_loss / len(data_loader), detailed_reports
