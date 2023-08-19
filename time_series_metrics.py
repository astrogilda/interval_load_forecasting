from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


class TimeSeriesMetrics:
    """Class to calculate metrics for time series data."""

    def __init__(self, y_true_file: str, y_pred_files: dict[str, str]) -> None:
        """Constructs all the necessary attributes for the TimeSeriesMetrics object.

        Parameters
        ----------
        y_true_file : str
            Path of the CSV file containing the true values.
        y_pred_files : dict[str, str]
            Dictionary containing the names of models, and the paths of the CSV files containing the predicted values.
        """
        self.y_true = self._read_file(y_true_file)
        self.y_pred_files = y_pred_files
        self.y_pred = {
            model: self._read_file(file)
            for model, file in y_pred_files.items()
        }
        # Check if y_true and y_pred_values are 2d arrays with the same shape
        if self.y_true.ndim != 2:
            raise ValueError("y_true must be a 2d array")
        if any(
            y_pred.shape != self.y_true.shape
            for y_pred in self.y_pred.values()
        ):
            raise ValueError("y_true and y_pred must have the same shape")

    def _read_file(self, file_name: str) -> np.ndarray:
        """Read data from CSV file.

        Parameters
        ----------
        file_name : str
            Path of the CSV file containing the data.

        Returns
        -------
        data : np.ndarray
            Array containing the data.
        """
        # Check if file exists
        if not Path(file_name).is_file():
            raise FileNotFoundError(f"File {file_name} not found")

        # Load data
        data = pd.read_csv(file_name, index_col=0).values
        return data

    def calculate_metrics(self):
        """Calculate metrics for the given true and predicted values.

        Returns
        -------
        metrics : pd.DataFrame
            Dataframe containing the metrics.
        """
        metrics = pd.DataFrame(
            columns=["MAE", "MAPE", "RMSE", "Bias"], index=self.y_pred.keys()
        )
        metrics.index.name = "Model"
        for model, y_pred in self.y_pred.items():
            metrics.loc[model, "MAE"] = mean_absolute_error(
                self.y_true, y_pred
            )
            metrics.loc[model, "MAPE"] = mean_absolute_percentage_error(
                self.y_true, y_pred
            )
            metrics.loc[model, "RMSE"] = mean_squared_error(
                self.y_true, y_pred, squared=False
            )
            metrics.loc[model, "Bias"] = np.mean(self.y_true - y_pred)
        return metrics

    def create_and_save_metrics(self, output_file: str) -> None:
        """Create and save metrics for the given true and predicted values.

        Parameters
        ----------
        output_file : str
            Path of the CSV file to save the metrics.
        """
        metrics = self.calculate_metrics()
        metrics.to_csv(output_file)
