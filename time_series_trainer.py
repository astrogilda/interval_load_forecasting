from functools import partial
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.forecasting.model_selection._split import BaseWindowSplitter

from common_constants import (
    CV_STRATEGY,
    HPO_FLAG,
    INITIAL_WINDOW_LENGTH,
    METRIC_NAME,
    MLFLOW_LOGGING_FLAG,
    MODEL_MAPPING,
    MODEL_NAME,
    MODEL_SPACES,
    OBJECTIVE_METRICS,
    OPTUNA_JOBS,
    OPTUNA_TRIALS,
    STEP_LENGTH,
    TARGET_VARIABLE,
    VAL_LENGTH,
    WINDOW_LENGTH,
)
from time_series_xy import TimeSeriesXy


class TimeSeriesTrainer:
    """
    Time series regression class.
    """

    def _calculate_and_log_shap_values(self, model, X_train, run_id: str):
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
        print(f"X_train shape: {X_train.shape}")
        print(f"X_train.head():\n{X_train.head()}")
        # Calculate SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        print(f"shap_values.shape: {shap_values.shape}")

        for step in range(
            shap_values.shape[2]
        ):  # Iterate over each forecast step
            # Isolate SHAP values for this step
            shap_values_step = shap_values[:, :, step]
            # Log SHAP values as a plot for this step
            shap_plot_path = f"shap_summary_step_{step}_{run_id}.png"
            shap.summary_plot(shap_values_step, X_train, show=False)
            plt.savefig(shap_plot_path)
            # Save SHAP values as a DataFrame
            shap_df_path = f"shap_summary_step_{step}_{run_id}.csv"
            shap_df = pd.DataFrame(
                shap_values_step.values, columns=X_train.columns
            )
            shap_df.to_csv(shap_df_path)

            if mlflow.active_run():
                # Log SHAP values as an artifact
                mlflow.log_artifact(shap_df_path)
                # Log SHAP values as a plot
                mlflow.log_artifact(shap_plot_path)

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
            Number of steps to take between each iteration during validation.
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
        hyperparameter_space = MODEL_SPACES[model_name]
        params = {}
        for param_name, distribution in hyperparameter_space.items():
            if isinstance(
                distribution, optuna.distributions.FloatDistribution
            ):
                params[param_name] = trial.suggest_float(
                    param_name, distribution.low, distribution.high
                )
            elif isinstance(
                distribution, optuna.distributions.IntDistribution
            ):
                params[param_name] = trial.suggest_int(
                    param_name, distribution.low, distribution.high
                )

        # Get the metric function from the dictionary
        metric_func = OBJECTIVE_METRICS[metric_name]
        # Create the model
        model_class = MODEL_MAPPING[model_name]
        model = model_class(**params)
        # Initialize the scores list
        scores = []

        # Iterate over the cross-validation splits
        for train_index, test_index in cv.split(df):
            # Split the data
            train, test = df.iloc[train_index], df.iloc[test_index]
            # Create X and y
            X_train, y_train = TimeSeriesXy.df_to_X_y(train, target_variable)
            X_test, y_test = TimeSeriesXy.df_to_X_y(test, target_variable)
            # Convert to numpy arrays
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
            # Create the scaler
            scaler = StandardScaler()
            # Fit the scaler on the training data and transform both training and test data
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # Fit the model
            if model_class == "xgb":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=10,
                    verbose=False,
                )
            else:
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
                mlflow.log_metric(metric_name, mean_score)  # type: ignore

        return mean_score  # type: ignore

    def train(
        self,
        df: pd.DataFrame,
        target_variable: str = TARGET_VARIABLE,
        model_name: str = MODEL_NAME,
        metric_name: str = METRIC_NAME,
        cv_strategy: str = CV_STRATEGY,
        hpo_flag: bool = HPO_FLAG,
        mlflow_logging_flag: bool = MLFLOW_LOGGING_FLAG,
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
        if model_name not in MODEL_MAPPING:
            raise ValueError(
                f"model_name must be one of: {list(MODEL_MAPPING.keys())}"
            )

        if metric_name not in OBJECTIVE_METRICS:
            raise ValueError(
                f"metric_name must be one of: {list(OBJECTIVE_METRICS.keys())}"
            )

        # Initialize cross-validator
        cv = self._create_cross_validator(cv_strategy)

        if mlflow_logging_flag:
            # Start the run
            mlflow.start_run(run_name=f"{model_name}_training")
            run_id = mlflow.active_run().info.run_id  # type: ignore
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
            print(f"Number of splits: {cv.get_n_splits(df)}")
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
                show_progress_bar=True,
                n_jobs=OPTUNA_JOBS,
            )

            # Best hyperparameters
            best_params = study.best_params
            best_score = study.best_value

            if mlflow.active_run():
                # Log parameters and metrics
                mlflow.log_params(best_params)
                mlflow.log_metric("best_hpo_score", best_score)

        model_class = MODEL_MAPPING[model_name]
        model = model_class(**best_params)
        X_train, y_train = TimeSeriesXy.df_to_X_y(df, target_variable)
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        # Calculate and log SHAP values
        self._calculate_and_log_shap_values(model, X_train, run_id)  # type: ignore

        if mlflow.active_run():
            # Log model
            mlflow.sklearn.log_model(model, f"model_{model_name}")
            # End the run
            mlflow.end_run()

        return model
