from typing import Any, Dict, List
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from models.base_scikitlearn_model import BaseModel


class LinearRegression:
    def __init__(self):
        """
        Scikit-learn Linear Regression model.
        """
        self.model = SklearnLinearRegression()

    def forward(self, batch: np.ndarray):
        """
        Predict function for the Linear Regression model.
        :param batch: Input batch.
        :return: Predictions.
        """
        return self.model.predict(batch)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the Linear Regression model.
        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        :param X: Input features.
        :return: Predictions.
        """
        return self.model.predict(X)


class BasicRegression(BaseModel):
    def __init__(self):
        super(BasicRegression, self).__init__()
        self.network = self._build_network()

    def _build_network(self) -> LinearRegression:
        """
        Build the scikit-learn Linear Regression network.
        """
        return LinearRegression()

    def calculate_detailed_report(self, predictions: List[np.ndarray], ground_truth: List[np.ndarray],
                                  **kwargs) -> Dict[str, Any]:
        """
        Calculate detailed report for the predictions.
        :param predictions: List of predictions.
        :param ground_truth: List of ground truth values.
        :return: Report as a dictionary.
        """
        mse = mean_squared_error(np.concatenate(ground_truth), np.concatenate(predictions))
        return {'mse': mse}

