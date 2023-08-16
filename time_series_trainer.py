from functools import partial
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
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

from common_constants import (
    CV_STRATEGY,
    HPO_FLAG,
    INITIAL_WINDOW_LENGTH,
    METRIC_NAME,
    MODEL_NAME,
    OPTUNA_TRIALS,
    STEP_LENGTH,
    TARGET_VARIABLE,
    VAL_LENGTH,
    WINDOW_LENGTH,
)
from time_series_xy import TimeSeriesXy


class TimeSeriesTrainer:
    """
    Time series regression and simulation class.

    Attributes
    ----------
    WINDOW_LENGTH : int
        Length of the sliding window for rolling cross-validation.
    INITIAL_WINDOW_LENGTH : int
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

    def _log_shap_values(self, model, X_train, run_id: str):
        """
        Log SHAP values as artifacts.

        Parameters
        ----------
        model : object
            Trained model object.
        X_train : pd.DataFrame
            Training data.
        run_id : str
            Unique identifier for the run.
        """
        # Calculate SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # Log SHAP values as a plot
        shap_plot_path = f"shap_summary_{run_id}.png"
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path)

        # Log SHAP values as a DataFrame (optional)
        shap_df_path = f"shap_values_{run_id}.csv"
        shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
        shap_df.to_csv(shap_df_path)
        mlflow.log_artifact(shap_df_path)

    def _create_cross_validator(
        self, cv_strategy: str = CV_STRATEGY
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
                window_length=WINDOW_LENGTH,
                start_with_window=True,
                step_length=STEP_LENGTH,
                fh=np.arange(1, VAL_LENGTH + 1),
            )
        elif cv_strategy == "expanding":
            cv = ExpandingWindowSplitter(
                initial_window=INITIAL_WINDOW_LENGTH,
                step_length=STEP_LENGTH,
                fh=np.arange(1, VAL_LENGTH + 1),
            )
        else:
            raise ValueError(
                "cv_strategy must be one of: 'rolling', 'expanding'"
            )
        return cv

    def _objective(
        self,
        trial,
        cv: BaseWindowSplitter,
        df: pd.DataFrame,
        target_variable: str,
        model_name: str,
        metric_name: str,
        log_flag: bool = False,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Parameters
        ----------
        trial : object
            Optuna trial object.
        df : pd.DataFrame
            DataFrame with y and weather data.
        target_variable : str
            Name of the target variable in df.
        model_name : str
            Name of the regression model to use. Must be one of: "rr", "rf", "xgb"
        metric_name : str
            Name of the metric to use for optimization. Must be one of: "mae", "rmse", "rmsle"
        cv : object
            Cross-validator object.
        log_flag : bool
            Flag to enable logging of parameters and metrics.

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
        # Initialize the scores list
        scores = []

        # Iterate over the cross-validation splits
        for train_index, test_index in cv.split(df):
            # Split the data
            train, test = df.iloc[train_index], df.iloc[test_index]
            # Create X and y
            X_train, y_train = TimeSeriesXy().df_to_X_y(train, target_variable)
            X_test, y_test = TimeSeriesXy().df_to_X_y(test, target_variable)
            # Convert to numpy arrays
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
            # Fit the model
            model.fit(X_train, y_train)
            # Return the metric
            score = metric_func(y_test, model.predict(X_test))
            scores.append(score)

        mean_score = np.mean(scores)

        if log_flag:
            # Log the parameters and metric for this trial
            with mlflow.start_run(
                run_name=f"{model_name}_trial_{trial.number}", nested=True
            ):
                mlflow.log_params(params)
                mlflow.log_metric(metric_name, mean_score)

        return mean_score

    def train(
        self,
        df: pd.DataFrame,
        target_variable: str = TARGET_VARIABLE,
        model_name: str = MODEL_NAME,
        metric_name: str = METRIC_NAME,
        cv_strategy: str = CV_STRATEGY,
        hpo_flag: bool = HPO_FLAG,
    ) -> Any:
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
        cv_strategy : str
            Cross-validation strategy. Must be one of: "rolling", "expanding".
        hpo_flag : bool
            Flag to enable hyperparameter optimization.

        Returns
        -------
        model : object
            Trained model object.
        """
        if model_name not in self.model_mapping:
            raise ValueError(
                f"model_name must be one of: {list(self.model_mapping.keys())}"
            )

        if metric_name not in self.objective_metrics:
            raise ValueError(
                f"metric_name must be one of: {list(self.objective_metrics.keys())}"
            )

        # Initialize cross-validator
        cv = self._create_cross_validator(cv_strategy)

        mlflow.start_run(run_name=f"{model_name}_training")
        run_id = mlflow.active_run().info.run_id
        # Log some basic information
        mlflow.log_params(
            {
                "model_name": model_name,
                "cv_strategy": cv_strategy,
                "window_length": WINDOW_LENGTH,
                "initial_window_length": INITIAL_WINDOW_LENGTH,
                "step_length": STEP_LENGTH,
                "metric_name": metric_name,
            }
        )

        best_params = {}
        if hpo_flag:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                partial(
                    self._objective,
                    cv=cv,
                    df=df,
                    target_variable=target_variable,
                    model_name=model_name,
                    metric_name=metric_name,
                ),
                n_trials=OPTUNA_TRIALS,
            )

            # Best hyperparameters
            best_params = study.best_params
            best_score = study.best_value

            # Log parameters and metrics
            mlflow.log_params(best_params)
            mlflow.log_metric("best_hpo_score", best_score)

        model_class = self.model_mapping[model_name]
        model = model_class(**best_params)
        X_train, y_train = TimeSeriesXy().df_to_X_y(df, target_variable)
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        # Log model
        mlflow.sklearn.log_model(model, f"model_{model_name}")

        # Log SHAP values
        self._log_shap_values(model, X_train, run_id)

        # End the run
        mlflow.end_run()

        return model

        """
        y_pred = model.predict(X_test.to_numpy())
        mae = mean_absolute_error(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)

        # Log SHAP values
        self._log_shap_values(model, X_train, f"shap_{model_name}")

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
        """
