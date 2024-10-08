from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class BaseModel:
    def __init__(self, lr: float = 0.001, **kwargs):
        """
        Basic regression model using scikit-learn.
        :param model_type: Type of regression model ('sgd' or 'ridge').
        :param lr: Learning rate.
        """
        self.lr = lr
        self.model = self._build_model(**kwargs)

    def _build_model(self):
        """
        Construct the regression model.
        """
        raise NotImplementedError()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, **kwargs):
        """
        Train the model using scikit-learn.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_val: Validation data.
        :param y_val: Validation labels.
        """
        self.model.fit(X_train, y_train)
        train_loss = mean_squared_error(y_train, self.model.predict(X_train))

        if X_val is not None and y_val is not None:
            val_loss = mean_squared_error(y_val, self.model.predict(X_val))
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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
