from pathlib import Path
from typing import Optional, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
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
from sktime.utils.plotting import plot_series
from xgboost import XGBRegressor

from timeseriesfeaturizer import TimeSeriesFeaturizer


class TimeSeriesForecaster:
    """
    Time series regression and simulation class.

    Attributes
    ----------
    WINDOW_LENGTH : int
        Length of the sliding window for rolling cross-validation.
    INITIAL_WINDOW : int
        Length of the initial window for expanding cross-validation.
    STEP_LENGTH : int
        Number of steps to take between each iteration in the walk-forward validation.
    MAX_LAGS : int
        Maximum number of lags to use for autoregressive features.
    objective_metrics : dict
        Objective metrics for Optuna optimization.
    model_spaces : dict
        Model hyperparameter spaces.
    model_mapping : dict
        Model names to model classes mapping.
    """

    WINDOW_LENGTH = 24 * 30 * 3  # 3 months
    INITIAL_WINDOW = 24 * 30 * 3  # 3 months
    STEP_LENGTH = 24 * 7  # 1 week
    MAX_LAGS = 3

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

    def objective(self, trial, X_train, y_train, model_name, metric_name):
        # Ensure that the model name is valid
        if model_name not in self.model_spaces:
            raise ValueError(f"Invalid model_name: {model_name}")

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

        model_class = self.model_mapping[model_name]
        model = model_class(**params)
        model.fit(X_train, y_train)
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
    ) -> object:
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

    def forecast(
        self,
        df: pd.DataFrame,
        target_variable: str,
        model_name: str,
        metric_name: str,
        step: int,
        cv_strategy: str = "rolling",
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
        for train_index, test_index in cv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            print(train.shape, test.shape)
            train = TimeSeriesFeaturizer.create_regression_data(
                train[target_variable],
                train.drop(target_variable, axis=1),
                ar_from_y=True,
                ar_from_weather_data=False,
                lags=step_length,
                max_lags=step_length,
            )
            test = TimeSeriesFeaturizer.create_regression_data(
                test[target_variable],
                test.drop(target_variable, axis=1),
                ar_from_y=True,
                ar_from_weather_data=False,
                lags=step_length,
                max_lags=step_length,
            )
            # print(train.shape, test.shape)
            # print(train.columns)
            # print(test.columns)

            X_train, y_train = (
                train.drop(target_variable, axis=1),
                train[target_variable],
            )
            X_test, y_test = (
                test.drop(target_variable, axis=1),
                test[target_variable],
            )

            # Optuna optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: self.objective(
                    trial,
                    X_train.to_numpy(),
                    y_train.to_numpy(),
                    model_name,
                    metric_name,
                ),
                n_trials=100,
            )

            # Best hyperparameters
            param_grid = study.best_params

            # Log parameters and metrics
            mlflow.log_params(param_grid)

            model_class = self.model_mapping[model_name]
            model = model_class(**param_grid)
            model.fit(X_train.to_numpy(), y_train.to_numpy())

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

    def simulate_production(
        self, initial_size: int, steps: int, cv_strategy: str = "rolling"
    ) -> None:
        """
        Simulates a production environment where new data comes in steps.

        Parameters
        ----------
        initial_size : int
            Initial size of the data.
        steps : int
            Number of steps to simulate in the production environment. This determines how many iterations the simulation will run.
        """
        for i in range(initial_size, initial_size + steps):
            # Merge y_data and weather_data
            if self.weather_data is None:
                df = self.y_data.iloc[: i + 1]
            else:
                df = pd.merge(
                    self.y_data.iloc[: i + 1],
                    self.weather_data.iloc[: i + 1],
                    left_index=True,
                    right_index=True,
                    how="inner",
                )

            df_results = self.forecast(
                df=df,
                target_variable=list(self.y_data)[0],
                model_name="rr",
                metric_name="mae",
                step=self.STEP_LENGTH,
                cv_strategy=cv_strategy,
            )

            self.save_simulation_results(df_results, f"results_step_{i}.csv")

            self.plot_simulation_results(f"results_step_{i}.csv")

    def save_simulation_results(
        self, df_results: pd.DataFrame, filename: str, folder: str = "results"
    ):
        """
        Saves the simulation results (actual and predicted values) to a CSV file.

        Parameters
        ----------
        actual : pd.Series
            Actual values.
        predicted : pd.Series
            Predicted values.
        filename : str
            Name of the file to save the results.
        folder : str
            Name of the folder to save the results.
        """
        # Create results folder if it doesn't exist
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)

        # Combine actual and predicted values into a DataFrame
        # df_results = pd.DataFrame({
        #    'actual': actual,
        #    'predicted': predicted
        # })

        # Save the DataFrame to a CSV file
        df_results.to_csv(Path(folder) / filename)

    def plot_simulation_results(
        self, filename: str, folder: str = "results"
    ) -> None:
        """
        Plots the simulation results (actual and predicted values).

        Parameters
        ----------
        filename : str
            Name of the file containing the results.
        folder : str
            Name of the folder containing the results.
        """
        # Load the results from the CSV file
        df_results = pd.read_csv(Path(folder) / filename, index_col=0)

        # Line plot of actual and predicted values
        plt.close("all")
        df_results.plot(
            kind="line", color=["blue", "red"], legend=True, figsize=(16, 8)
        )
        plt.title("Hourly Electricity Load", fontsize=20)
        plt.xlabel("Time", fontsize=15)
        plt.ylabel("Electricity Load", fontsize=15)
        plt.savefig(
            "pred_vs_true_lineplot.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # Calculate residuals (actual - predicted)
        df_results["residuals"] = (
            df_results["actual"] - df_results["predicted"]
        )

        print(type(df_results))

        # Add necessary time attributes
        df_results.index = pd.to_datetime(df_results.index)
        df_results["hour"] = df_results.index.hour.values
        df_results["day"] = df_results.index.dayofweek.values
        df_results["month"] = df_results.index.month.values

        # Scatter plot of residuals vs. hour of day
        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        plt.scatter(df_results["actual"], df_results["predicted"], alpha=0.5)
        plt.plot(
            [df_results["actual"].min(), df_results["actual"].max()],
            [df_results["actual"].min(), df_results["actual"].max()],
            "k--",
            label="Perfect Fit",
        )
        plt.xlabel("True Load", fontsize=15)
        plt.ylabel("Predicted Load", fontsize=15)
        plt.legend()
        plt.title("Actual vs. Predicted Load", fontsize=20)
        plt.savefig(
            "pred_vs_true_scatterplot.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # Histogram of residuals
        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        df_results["residuals"].hist(bins=30, edgecolor="k")
        plt.xlabel("Residual", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.title("Histogram of Residuals", fontsize=20)
        plt.savefig(
            "residuals_histogram.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # Line plot of residuals over time
        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        plt.plot(
            df_results.index,
            df_results["residuals"],
            label="Residuals",
            color="red",
        )
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Datetime", fontsize=15)
        plt.ylabel("Residual", fontsize=15)
        plt.title("Residuals Over Time", fontsize=20)
        plt.savefig(
            "residuals_lineplot.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # Group by hour and day and calculate mean residuals
        grouped = (
            df_results.groupby(["hour", "day"])["residuals"]
            .mean()
            .reset_index()
        )
        grouped = grouped.pivot(
            index="hour", columns="day", values="residuals"
        )

        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        sns.heatmap(grouped, cmap="coolwarm", annot=True, fmt=".2f")
        plt.title("Mean Residuals by Hour and Day of Week", fontsize=20)
        plt.xlabel("Day of Week", fontsize=15)
        plt.ylabel("Hour of Day", fontsize=15)
        plt.savefig(
            "residuals_heatmap.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


"""
if __name__ == "__main__":

from timeseriesregression import TimeSeriesRegression
# Paths to input CSV files
y_file = "data/load.csv"
weather_data_file = None

# Create TimeSeriesRegression object
tsr = TimeSeriesRegression(y_file, weather_data_file)
initial_size = tsr.y_data.shape[0]
steps=1
tsr.simulate_production(initial_size, steps)


"""
