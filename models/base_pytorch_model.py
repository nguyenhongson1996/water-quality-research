from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class BaseModel:
    def __init__(self, model_type: str = 'sgd', lr: float = 0.001):
        """
        Basic regression model using scikit-learn.
        :param input_dim: Input dimension (used for compatibility, not necessary for scikit-learn).
        :param model_type: Type of regression model ('sgd' or 'ridge').
        :param lr: Learning rate.
        """
        self.lr = lr
        self.model = self._build_model(model_type)

    def _build_model(self, model_type: str):
        """
        Construct the regression model.
        """
        if model_type == 'sgd':
            return SGDRegressor(learning_rate='constant', eta0=self.lr)
        elif model_type == 'ridge':
            return Ridge(alpha=self.lr)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, epochs: int = 100, patience: int = 10):
        """
        Train the model using scikit-learn.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_val: Validation data.
        :param y_val: Validation labels.
        :param epochs: Number of training epochs.
        :param patience: Number of epochs to wait for improvement before early stopping.
        """
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.partial_fit(X_train, y_train)  # Incremental training

            train_loss = mean_squared_error(y_train, self.model.predict(X_train))

            if X_val is not None and y_val is not None:
                val_loss = mean_squared_error(y_val, self.model.predict(X_val))
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    break
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.
        """
        return self.model.predict(X)

    @abstractmethod
    def calculate_detailed_report(self, predictions: np.ndarray, ground_truth: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Calculate the detailed report. This method should be overridden.
        """
        raise NotImplementedError()
