from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from xgboost import XGBRegressor


class TimeSeriesModel:
    """
    Class for time series modeling, including hyperparameter tuning and forecasting.

    Attributes
    ----------
    model_mapping : dict
        Model names to model classes mapping.
    model_spaces : dict
        Hyperparameter spaces for different models.
    objective_metrics : dict
        Metrics for Optuna objective function.
    """

    model_mapping = {
        "rr": Ridge,
        "rf": RandomForestRegressor,
        "xgb": XGBRegressor,
    }

    model_spaces = {
        "rr": {"alpha": optuna.distributions.FloatDistribution(0.0, 1.0)},
        "rf": {
            "n_estimators": optuna.distributions.IntDistribution(2, 150),
            "max_depth": optuna.distributions.IntDistribution(1, 32),
        },
        "xgb": {
            "n_estimators": optuna.distributions.IntDistribution(2, 150),
            "max_depth": optuna.distributions.IntDistribution(1, 10),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
        },
    }

    objective_metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "mape": mean_absolute_percentage_error,
    }

    def __init__(self) -> None:
        pass

    def objective(
        self,
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        metric_name: str,
    ) -> float:
        """
        Objective function for Optuna hyperparameter tuning.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object.
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training target.
        model_name : str
            Name of the model.
        metric_name : str
            Name of the metric.

        Returns
        -------
        float
            Metric value.
        """
        if model_name not in self.model_spaces:
            raise ValueError(f"Invalid model_name: {model_name}")

        hyperparameter_space = self.model_spaces[model_name]
        params = {
            param_name: trial.suggest_float(
                param_name, distribution.low, distribution.high
            )
            if isinstance(
                distribution, optuna.distributions.UniformDistribution
            )
            else trial.suggest_int(
                param_name, distribution.low, distribution.high
            )
            for param_name, distribution in hyperparameter_space.items()
        }

        metric_func = self.objective_metrics[metric_name]
        model_class = self.model_mapping[model_name]
        model = model_class(**params)
        model.fit(X_train, y_train)
        return metric_func(y_train, model.predict(X_train))

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        metric_name: str,
    ) -> Any:
        """
        Train a model using Optuna for hyperparameter tuning.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training target.
        model_name : str
            Name of the model.
        metric_name : str
            Name of the metric.

        Returns
        -------
        Any
            Trained model.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(
                trial, X_train, y_train, model_name, metric_name
            ),
            n_trials=100,
        )
        best_params = study.best_params
        model_class = self.model_mapping[model_name]
        model = model_class(**best_params)
        model.fit(X_train, y_train)
        return model

    def forecast(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """
        Forecast using a trained model.

        Parameters
        ----------
        model : Any
            Trained model.
        X_test : np.ndarray
            Test features.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        return model.predict(X_test)
