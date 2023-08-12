import os
from typing import Optional, Tuple, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.utils.plotting import plot_series
from statsmodels.tsa.stattools import pacf
from xgboost import XGBRegressor

FIGSIZE = (16, 8)


class TimeSeriesRegression:
    """
    Time series regression and simulation class.

    Attributes
    ----------
    y_data : pd.Series
        Time series data.
    weather_data : pd.DataFrame
        Exogenous weather data.
    model_mapping : dict
        Model names to model classes mapping.
    """

    WINDOW_LENGTH = 24 * 30 * 3  # 3 months
    INITIAL_WINDOW = 24 * 30 * 3  # 3 months
    STEP_LENGTH = 24 * 7  # 1 week
    MAX_LAGS = 3

    # Define metrics for training
    training_metrics = {"mae": mean_absolute_error, "mse": mean_squared_error}

    # Define metrics for Optuna objective function
    objective_metrics = {"mae": mean_absolute_error, "mse": mean_squared_error}

    # Define model mapping
    model_mapping = {
        "rr": Ridge,
        "rf": RandomForestRegressor,
        "xgb": XGBRegressor,
    }

    # Define model hyperparameter spaces
    model_spaces = {
        "rr": {"alpha": optuna.distributions.UniformDistribution(0.0, 1.0)},
        "rf": {
            "n_estimators": optuna.distributions.IntUniformDistribution(
                2, 150
            ),
            "max_depth": optuna.distributions.IntUniformDistribution(1, 32),
        },
        "xgb": {
            "n_estimators": optuna.distributions.IntUniformDistribution(
                2, 150
            ),
            "max_depth": optuna.distributions.IntUniformDistribution(1, 10),
            "learning_rate": optuna.distributions.UniformDistribution(
                0.01, 0.3
            ),
        },
    }

    def __init__(
        self, y_file: str, weather_data_file: Optional[str] = None
    ) -> None:
        """
        Constructs all the necessary attributes for the TimeSeriesRegression object.

        Parameters
        ----------
        y_file : str
            Path of the CSV file containing y data.
        weather_data_file : str
            Path of the CSV file containing weather data.
        """
        self.y_data, self.weather_data = self._load_data(
            y_file
        ), self._load_data(weather_data_file)

    @staticmethod
    def _load_data(file_name: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Load data from CSV files and perform sanity checks.

        Parameters
        ----------
        file_name : str
            Path of the CSV file containing the data.

        Returns
        -------
        data : pd.DataFrame
            Dataframe containing the data.
        """
        # If file_name is None, return None
        if file_name is None:
            return None

        # Check if file exists
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} not found")

        # Load data
        data = pd.read_csv(file_name, index_col=0)
        # Rename index to "DateTime"
        data.index.rename("DateTime", inplace=True)
        # Convert the index to datetime type
        data.index = pd.to_datetime(data.index)

        # Check if data is a pd.DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check for missing values and forward fill
        data = data.fillna(method="ffill")

        return data

    @staticmethod
    def _handle_duplicate_indices(
        *dfs: pd.DataFrame,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        """
        Averages rows with duplicate indices in one or more pandas DataFrames.

        Parameters
        ----------
        dfs : tuple of pd.DataFrame
            One or more pandas DataFrames with timestamps as index.

        Returns
        -------
        output_dfs : pd.DataFrame or tuple of pd.DataFrame
            DataFrame(s) with unique indices. If a single DataFrame was passed as input, a single DataFrame is returned.
            If multiple DataFrames were passed, a tuple of DataFrames is returned.
        """
        output_dfs = []
        for df in dfs:
            # Check if df is a pandas DataFrame or Series
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    "All inputs must be pandas DataFrames or Series"
                )

            # Group by index and average rows with duplicate indices
            df = df.groupby(df.index).mean()
            output_dfs.append(df)

        if len(output_dfs) == 1:
            return output_dfs[0]  # return single dataframe
        else:
            return tuple(output_dfs)  # return tuple of dataframes

    @staticmethod
    def _find_best_lag_pacf(y: pd.Series, max_lag: int) -> int:
        """
        Finds the best lag for a time series using PACF.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        max_lag : int
            Maximum number of lags to consider.

        Returns
        -------
        best_lag : int
            Lag with the highest PACF.
        """
        # Calculate PACF
        lag_pacf = pacf(y, nlags=max_lag)

        # Find the lag with the highest PACF (ignoring lag 0)
        best_lag = np.argmax(lag_pacf[1:]) + 1

        return best_lag

    @staticmethod
    def _align_timestamps(
        df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.Series, pd.Series]:
        """
        Aligns the timestamps of two dataframes, in case they have different frequencies or missing time points.

        Parameters
        ----------
        df1 : pd.DataFrame
            First DataFrame.
        df2 : pd.DataFrame
            Second DataFrame.

        Returns
        -------
        df1_aligned, df2_aligned : tuple of pd.DataFrame
            Aligned dataframes.
        """
        # Ensure the DataFrames are sorted by the index
        df1.sort_index(inplace=True)
        df2.sort_index(inplace=True)

        # Ensure the indices are unique
        df1, df2 = TimeSeriesRegression._handle_duplicate_indices(df1, df2)

        # Reindex the DataFrames to match the union of the two indices and forward fill any missing data
        common_index = df1.index.intersection(df2.index)
        df1_aligned = df1.reindex(common_index, method="ffill")
        df2_aligned = df2.reindex(common_index, method="ffill")

        return df1_aligned, df2_aligned

    @staticmethod
    def _create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates calendar features from a DataFrame with a DateTime index.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTime index.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with calendar features.
        """
        # Ensure that index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Extract calendar features from the index
        df["year"] = df.index.year
        df["month"] = df.index.month / 12  # normalize to [0, 1]
        df["day"] = df.index.day / 31  # normalize to [0, 1]
        df["dayofweek"] = df.index.dayofweek / 7  # normalize to [0, 1]
        df["hour"] = df.index.hour / 24  # normalize to [0, 1]
        df["minute"] = df.index.minute / 60  # normalize to [0, 1]

        return df

    @staticmethod
    def _create_regression_data(
        y: pd.DataFrame,
        weather_data: Optional[pd.DataFrame] = None,
        ar_from_y: bool = True,
        ar_from_weather_data: bool = False,
        lags: int = 3,
        max_lags: int = 3,
        use_pacf: bool = False,
    ) -> pd.DataFrame:
        """
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series data.
        weather_data : pd.DataFrame
            Exogenous weather data.
        ar_from_y : bool
            If True, add autoregressive features from y.
        ar_from_weather_data : bool
            If True, add autoregressive features from weather data.
        lags : int
            Number of autoregressive features to add.
        max_lags : int
            Maximum number of lags possible; this is a function of step_length in the cross-validator.

        Returns
        -------
        df : pd.DataFrame
            DataFrame ready for regression, with weather data and (if requested) autoregressive features.
        """
        # Ensure lags is not greater than max_lags
        if lags > max_lags:
            raise ValueError(
                f"lags cannot be greater than max_lags ({max_lags})"
            )

        # Merge y with weather data
        if weather_data is None:
            df = y.copy()
        else:
            df = pd.merge(
                y, weather_data, left_index=True, right_index=True, how="inner"
            )

        # Create calendar features
        df = TimeSeriesRegression._create_calendar_features(df)

        # If autoregressive features from y are requested, add them
        if ar_from_y:
            if use_pacf:
                best_lag_y = TimeSeriesRegression._find_best_lag_pacf(
                    y, max_lags
                )
            else:
                best_lag_y = lags
            for i in range(1, best_lag_y + 1):
                df[f"y_lag_{i}"] = df[y.name].shift(i)

        # If autoregressive features from weather_data are requested, add them
        if weather_data is not None and ar_from_weather_data:
            for column in weather_data.columns:
                if use_pacf:
                    best_lag_column = TimeSeriesRegression._find_best_lag_pacf(
                        df[column], max_lags
                    )
                else:
                    best_lag_column = lags
                for i in range(1, best_lag_column + 1):
                    df[f"{column}_lag_{i}"] = df[column].shift(i)

        return df.dropna()

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
        # shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
        # shap_df.to_csv("shap_values.csv")
        # mlflow.log_artifact("shap_values.csv", artifact_name)

    def _walk_forward_forecasting(
        self,
        df: pd.DataFrame,
        target_variable: str,
        model_name: str,
        metric_name: str,
        param_grid: dict,
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
            Name of the regression model to use. Must be one of: 'ar', 'rf', 'lr', 'gbr'.
        param_grid : dict
            Parameter grid for the specified model.
        step : int
            Number of steps to forecast. This determines the forecast horizon for each iteration in the walk-forward validation.

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
            train = self._create_regression_data(
                train[target_variable],
                train.drop(target_variable, axis=1),
                ar_from_y=True,
                ar_from_weather_data=False,
                lags=step_length,
                max_lags=step_length,
            )
            test = self._create_regression_data(
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
        # mlflow.sklearn.save_model(model, "best_model")

        # Add tags or notes (optional)
        # mlflow.set_tags({"description": "Time series forecasting with walk-forward validation"})

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
        param_grid = {"fit_intercept": True}  # , 'normalize': False}

        if self.y_data is None:
            raise ValueError(
                "y_data and must be specified before calling simulate_production()."
            )

        # Align timestamps if necessary
        if self.weather_data is not None:
            self.y_data, self.weather_data = self._align_timestamps(
                self.y_data, self.weather_data
            )

        for i in range(initial_size, initial_size + steps):
            # Merge y_data and weather_data
            df = pd.merge(
                self.y_data.iloc[: i + 1],
                self.weather_data.iloc[: i + 1],
                left_index=True,
                right_index=True,
                how="inner",
            )

            df_results = self._walk_forward_forecasting(
                df,
                target_variable=list(self.y_data)[0],
                model_name="lr",
                param_grid=param_grid,
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
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Combine actual and predicted values into a DataFrame
        # df_results = pd.DataFrame({
        #    'actual': actual,
        #    'predicted': predicted
        # })

        # Save the DataFrame to a CSV file
        df_results.to_csv(os.path.join(folder, filename))

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
        df_results = pd.read_csv(os.path.join(folder, filename), index_col=0)

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
        plt.figure(figsize=FIGSIZE)
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
        plt.figure(figsize=FIGSIZE)
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
        plt.figure(figsize=FIGSIZE)
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
        plt.figure(figsize=FIGSIZE)
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
