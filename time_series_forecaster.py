from typing import Optional, Tuple, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.forecasting.model_selection._split import BaseWindowSplitter
from sktime.utils.plotting import plot_series
from xgboost import XGBRegressor

from time_series_featurizer import TimeSeriesFeaturizer


class TimeSeriesForecaster:
    """
    Time series regression and simulation class.

    Attributes
    ----------
    WINDOW_LENGTH : int
        Length of the sliding window for rolling cross-validation.
    INITIAL_WINDOW : int
        Length of the initial window for expanding cross-validation.
    MAX_LAGS : int
        Maximum number of lags to use for autoregressive features.
    OPTUNA_TRIALS : int
        Number of trials for Optuna optimization.
    objective_metrics : dict
        Objective metrics for Optuna optimization.
    model_spaces : dict
        Model hyperparameter spaces.
    model_mapping : dict
        Model names to model classes mapping.
    """

    WINDOW_LENGTH = 24 * 30 * 3  # 3 months
    INITIAL_WINDOW = 24 * 30 * 3  # 3 months
    OPTUNA_TRIALS = 100
    AR_FROM_Y = True  # Autoregressive features from y
    AR_FROM_WEATHER_DATA = False  # Autoregressive features from weather data
    LAGS = 3  # Number of lags to use for autoregressive features
    MAX_LAGS = 3  # Maximum number of lags to use for autoregressive features
    HPO_FLAG = False  # Flag to enable hyperparameter optimization
    CV_STRATEGY = "rolling"  # Cross-validation strategy. Must be one of: "rolling", "expanding"

    # Define metrics for Optuna objective function
    objective_metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "mape": mean_absolute_percentage_error,
    }

    # Define model mapping
    model_mapping = {
        "rr": Ridge,
        "rf": RandomForestRegressor,
        "xgb": XGBRegressor,
    }

    # Define model hyperparameter spaces
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

    def __init__(
        self, y_data: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Constructs all the necessary attributes for the TimeSeriesRegression object.

        Parameters
        ----------
        y_data : pd.DataFrame
            Time series data.
        weather_data : str
            Exogenous weather data.
        """
        self.y_data = y_data
        self.weather_data = weather_data

        if self.y_data is None:
            raise ValueError(
                "y_data and must be specified before calling simulate_production()."
            )

    def objective(
        self,
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        metric_name: str,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Parameters
        ----------
        trial : object
            Optuna trial object.
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training labels.
        model_name : str
            Name of the regression model to use. Must be one of: "rr", "rf", "xgb"
        metric_name : str
            Name of the metric to use for optimization. Must be one of: "mae", "rmse", "rmsle"

        Returns
        -------
        metric_func : float
            Metric value.
        """
        # Define hyperparameters to be optimized based on the model name
        hyperparameter_space = self.model_spaces[model_name]
        params = {}
        for param_name, distribution in hyperparameter_space.items():
            if isinstance(
                distribution, optuna.distributions.UniformDistribution
            ):
                params[param_name] = trial.suggest_float(
                    param_name, distribution.low, distribution.high
                )
            elif isinstance(
                distribution, optuna.distributions.IntUniformDistribution
            ):
                params[param_name] = trial.suggest_int(
                    param_name, distribution.low, distribution.high
                )

        # Get the metric function from the dictionary
        metric_func = self.objective_metrics[metric_name]
        # Create the model
        model_class = self.model_mapping[model_name]
        model = model_class(**params)
        # Fit the model
        model.fit(X_train, y_train)
        # Return the metric
        return metric_func(y_train, model.predict(X_train))

    def log_shap_values(self, model, X_train, artifact_name):
        # Calculate SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # Log SHAP values as a plot
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png", artifact_name)

        # Log SHAP values as a DataFrame (optional)
        shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
        shap_df.to_csv("shap_values.csv")
        mlflow.log_artifact("shap_values.csv", artifact_name)

    def create_cross_validator(
        self,
        cv_strategy: str,
        step_length: int,
        fh: Union[list[int], np.ndarray],
    ) -> BaseWindowSplitter:
        """
        Creates a cross-validator object.

        Parameters
        ----------
        cv_strategy : str
            Cross-validation strategy. Must be one of: "rolling", "expanding".
        step_length : int
            Number of steps to take between each iteration in the walk-forward validation.
        fh : Union[list[int], np.ndarray]
            Forecast horizon.

        Returns
        -------
        cv : object
            Cross-validator object.
        """
        if cv_strategy == "rolling":
            cv = SlidingWindowSplitter(
                window_length=self.WINDOW_LENGTH,
                start_with_window=True,
                step_length=step_length,
                fh=fh,
            )
        elif cv_strategy == "expanding":
            cv = ExpandingWindowSplitter(
                initial_window=self.INITIAL_WINDOW,
                step_length=step_length,
                fh=fh,
            )
        else:
            raise ValueError(
                "cv_strategy must be one of: 'rolling', 'expanding'"
            )
        return cv

    def create_regression_data(
        self, df: pd.DataFrame, target_variable: str, fh: int
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
        """
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with a DateTime index, containing both y and optional weather data.
        target_variable: str
            Name of the target (y) variable.
        fh : int
            Forecast horizon.

        Returns
        -------
        X : pd.DataFrame
            DataFrame ready for regression, with optional weather data and (if requested) autoregressive features.
        y : pd.DataFrame
            DataFrame with the target variable.
        """
        # Create X and y
        X = df.drop(columns=target_variable)
        y = df[target_variable].shift(-(fh - 1)).dropna()

        # Drop NaN values from y and corresponding rows from X
        nan_rows = y.isna()
        y = y.dropna()
        X = X.loc[~nan_rows]

        return X, y

    def forecast(
        self,
        df: pd.DataFrame,
        target_variable: str,
        model_name: str,
        metric_name: str,
        step: int,
        cv_strategy: str = CV_STRATEGY,
        hpo_flag: bool = HPO_FLAG,
    ) -> pd.DataFrame:
        """
        Perform walk forward forecasting using a specified regression model and parameter grid.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with y and weather data.
        target_variable : str
            Name of the target variable in df.
        model_name : str
            Name of the regression model to use. Must be one of: "rr", "rf", "xgb"
        metric_name : str
            Name of the metric to use for optimization. Must be one of: "mae", "rmse", "rmsle"
        step : int
            Number of steps to forecast. This determines the forecast horizon for each iteration in the walk-forward validation.
        cv_strategy : str
            Cross-validation strategy. Must be one of: "rolling", "expanding".
        hpo_flag : bool
            Flag to enable hyperparameter optimization.

        Returns
        -------
        y_true_all, y_pred_all : tuple of pd.Series
            True and predicted values.
        """
        if model_name not in self.model_mapping:
            raise ValueError(
                f"model_name must be one of: {list(self.model_mapping.keys())}"
            )

        if metric_name not in self.objective_metrics:
            raise ValueError(
                f"metric_name must be one of: {list(self.objective_metrics.keys())}"
            )

        # fh is the forecast horizon, i.e. the number of steps to forecast. step_length is the number of steps to take between each iteration.
        step_length = step + 1
        fh = np.arange(1, step + 1)

        # Initialize cross-validator
        cv = self.create_cross_validator(cv_strategy, step_length, fh)

        y_true_all, y_pred_all, indices_all = [], [], []

        mlflow.start_run()
        # Log some basic information
        mlflow.log_params(
            {
                "model_name": model_name,
                "cv_strategy": cv_strategy,
                "step": step,
                "metric_name": metric_name,
            }
        )

        def featurizer(x):
            return TimeSeriesFeaturizer.create_features(
                x,
                target_variable,
                ar_from_y=self.AR_FROM_Y,
                ar_from_weather_data=self.AR_FROM_WEATHER_DATA,
                lags=step_length or self.LAGS,
                max_lags=step_length or self.MAX_LAGS,
            )

        for train_index, test_index in cv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            print(train.shape, test.shape)

            train, _ = featurizer(train)
            test, _ = featurizer(test)

            X_train, y_train = self.create_regression_data(
                train, target_variable, step
            )
            X_test, y_test = self.create_regression_data(
                test, target_variable, step
            )

            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()

            # Optuna optimization
            param_grid = {}
            if hpo_flag:
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial, X_train=X_train, y_train=y_train: self.objective(
                        trial,
                        X_train,
                        y_train,
                        model_name,
                        metric_name,
                    ),
                    n_trials=self.OPTUNA_TRIALS,
                )

                # Best hyperparameters
                param_grid = study.best_params

                # Log parameters and metrics
                mlflow.log_params(param_grid)

            model_class = self.model_mapping[model_name]
            model = model_class(**param_grid)
            model.fit(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(model, f"model_{model_name}")

            y_pred = model.predict(X_test.to_numpy())
            mae = mean_absolute_error(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mae", mae)

            # Log SHAP values
            self.log_shap_values(model, X_train, f"shap_{model_name}")

            y_true_all.extend(list(y_test))
            y_pred_all.extend(list(y_pred))
            indices_all.extend(list(X_test.index))

        mae = mean_absolute_error(y_true_all, y_pred_all)
        print(f"Mean Absolute Error: {mae}")

        # Log overall metrics
        mlflow.log_metric("overall_mae", mae)

        # Save the best model (optional)
        mlflow.sklearn.save_model(model, "best_model")

        # Add tags or notes (optional)
        mlflow.set_tags(
            {
                "description": "Time series forecasting with walk-forward validation"
            }
        )

        mlflow.end_run()

        plot_series(
            pd.Series(y_true_all, name="y_true"),
            pd.Series(y_pred_all, name="y_pred"),
        )

        # return pd.Series(y_true_all, name='y_true'), pd.Series(y_pred_all, name='y_pred')

        # Combine actual and predicted values into a DataFrame
        df_results = pd.DataFrame(
            {"actual": y_true_all, "predicted": y_pred_all}
        )
        df_results.index = indices_all
        return df_results
