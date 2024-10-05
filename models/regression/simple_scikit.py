from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from models.base_scikitlearn_model import BaseModel


class SimpleSkLearnRegression(BaseModel):
    def __init__(self):
        super(SimpleSkLearnRegression, self).__init__()
        self.network = self._build_model()

    def _build_model(self) -> LinearRegression:
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


class SimpleSVR(BaseModel):
    def __init__(self):
        super(SimpleSVR, self).__init__()

    def _build_model(self) -> SVR:
        return SVR()
