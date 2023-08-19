from pathlib import Path
from typing import Any, Tuple

import mlflow
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common_constants import (
    FORECAST_HORIZON,
    HPO_FLAG,
    INITIAL_TRAIN_LENGTH,
    LAGS,
    METRIC_NAME,
    MLFLOW_LOGGING_FLAG,
    MODEL_NAME,
    OBJECTIVE_METRICS,
    TARGET_VARIABLE,
    TEST_LENGTH,
)
from time_series_trainer import TimeSeriesTrainer
from time_series_xy import TimeSeriesXy


class TimeSeriesSimulator:
    """
    Simulates a production environment where new data comes in steps.

    Attributes
    ----------
    df_test : pd.DataFrame
        DataFrame containing the test data.
    df_test_pred : pd.DataFrame
        DataFrame containing the predicted values.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Constructs all the necessary attributes for the TimeSeriesRegression object.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the combined y and weather data.
        """
        self.df = df
        self.df_test = pd.DataFrame()
        self.df_test_pred = pd.DataFrame()
        self.score = None

    @staticmethod
    def forecast(
        df: pd.DataFrame, target_variable: str, model: Any, scaler: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Forecast using the trained model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the combined y and weather data.
        target_variable : str
            Target variable name.
        model : Any
            Trained model.

        Returns
        -------
        y_test : pd.DataFrame
            DataFrame containing the actual values.
        y_test_pred : pd.DataFrame
            DataFrame containing the predicted values.
        """
        X_test, y_test = TimeSeriesXy.df_to_X_y(df, target_variable)
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = model.predict(X_test_scaled)
        y_test_pred = pd.DataFrame(
            y_test_pred, index=y_test.index, columns=y_test.columns
        )

        return y_test, y_test_pred

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to remove duplicates from.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with duplicates removed.
        """
        df = df.loc[~df.index.duplicated(keep="first")]
        return df

    def simulate_production(
        self,
        initial_size: int = INITIAL_TRAIN_LENGTH,
        test_length: int = TEST_LENGTH,
        metric_name: str = METRIC_NAME,
    ) -> None:
        """
        Simulates a production environment where new data comes in steps.

        Parameters
        ----------
        initial_size : int
            Initial size of the data.
        test_length : int
            Number of steps to simulate in the production environment. This determines how many iterations the simulation will run.
        """
        i = initial_size
        while True:
            # Merge y_data and weather_data
            df_train, df_test = (
                self.df.iloc[:i, :],
                self.df.iloc[i : i + test_length, :],
            )

            # Break if there is no more data to simulate
            # print(f"df_test.shape: {df_test.shape}")
            if df_test.shape[0] < LAGS + FORECAST_HORIZON + 1:
                break

            # Train model
            trainer = TimeSeriesTrainer()
            trained_model, fitted_scaler = trainer.train(df=df_train)

            # Forecast
            df_test, df_test_pred = TimeSeriesSimulator.forecast(
                df_test, TARGET_VARIABLE, trained_model, fitted_scaler
            )
            # print(f"df_test.shape: {df_test.shape}")
            # print(f"df_test_pred.shape: {df_test_pred.shape}")
            # print("")

            # Update simulation results
            self.update_simulation_results(
                metric_name,
                df_test,
                df_test_pred,
            )

            # Update i
            i = i + LAGS + FORECAST_HORIZON

        # Remove duplicates
        self.df_test = TimeSeriesSimulator.remove_duplicates(self.df_test)
        self.df_test_pred = TimeSeriesSimulator.remove_duplicates(
            self.df_test_pred
        )

        metric_func = OBJECTIVE_METRICS[metric_name]
        overall_score = metric_func(self.df_test, self.df_test_pred)
        if MLFLOW_LOGGING_FLAG:
            # Log overall metrics
            mlflow.log_metric(f"overall_{metric_name}_score", overall_score)

        # Save simulation results
        self.save_simulation_results(
            self.df_test, "actual.csv", folder="results"
        )
        pred_filename = (
            f"predicted_model={MODEL_NAME}_hpometric={METRIC_NAME}.csv"
            if HPO_FLAG
            else f"predicted_model={MODEL_NAME}.csv"
        )
        self.save_simulation_results(
            self.df_test_pred, pred_filename, folder="results"
        )

        # Save scores
        self.score = overall_score

        # Save residual plots
        TimeSeriesSimulator.plot_simulation_results(
            filename_actual="actual.csv",
            filename_pred=pred_filename,
            folder="results/residuals",
        )

    def update_simulation_results(
        self,
        metric_name,
        df_test,
        df_test_pred,
    ) -> None:
        """
        Update simulation results and log metrics.

        Parameters
        ----------
        metric_name : str
            Metric name.
        df_test_pred : pd.DataFrame
            DataFrame containing the test data, including the actual values and the predicted values.
        """
        metric_func = OBJECTIVE_METRICS[metric_name]

        score = metric_func(df_test, df_test_pred)

        if MLFLOW_LOGGING_FLAG:
            # Log metrics for this simulation run
            with mlflow.start_run(run_name="simulation_run", nested=True):
                mlflow.log_metric(f"{metric_name}_score", score)

        # Combine actual and predicted values into a DataFrame
        self.df_test = pd.concat([self.df_test, df_test])
        self.df_test_pred = pd.concat([self.df_test_pred, df_test_pred])

    def save_simulation_results(
        self, df_results: pd.DataFrame, filename: str, folder: str = "results"
    ) -> None:
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

        # Save the DataFrame to a CSV file
        df_results.to_csv(Path(folder) / filename)

    @staticmethod
    def plot_simulation_results(
        filename_pred: str,
        filename_actual: str,
        folder: str = "figures/results",
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
        df_actual = pd.read_csv(Path(folder) / filename_actual, index_col=0)
        df_pred = pd.read_csv(Path(folder) / filename_pred, index_col=0)

        """
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
        """

        # Calculate residuals (actual - predicted)
        df_residuals = df_actual - df_pred

        # Add necessary time attributes
        df_residuals.index = pd.to_datetime(df_residuals.index)
        df_residuals["hour"] = df_residuals.index.hour.values
        df_residuals["day"] = df_residuals.index.dayofweek.values
        df_residuals["month"] = df_residuals.index.month.values

        for column in df_actual.columns:
            plt.close("all")
            plt.scatter(
                df_actual[column],
                df_pred[column],
                alpha=0.5,
                color="blue",
                label=f"Actual vs. Predicted {column}",
            )
            plt.plot(
                [df_actual[column].min(), df_actual[column].max()],
                [df_actual[column].min(), df_actual[column].max()],
                "white",
                ls="--",
                lw=1,
                label="Perfect Fit",
            )
            plt.xlabel("True Load", fontsize=15)
            plt.ylabel("Predicted Load", fontsize=15)
            plt.legend()
            plt.title(f"Actual vs. Predicted {column}", fontsize=20)
            plt.savefig(
                f"figures/actual_vs_predicted_{column}_scatterplot.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1,
            )

        # Histogram of mean residuals
        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        df_residuals.mean(axis=1).hist(bins=30, edgecolor="k")
        plt.xlabel("Residual", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.title("Histogram of Mean Residuals", fontsize=20)
        plt.savefig(
            "figures/residuals_histogram_mean.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # Histogram of .95 quantile residuals
        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        df_residuals.quantile(0.95, axis=1).hist(bins=30, edgecolor="k")
        plt.xlabel("Residual", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.title("Histogram of max Residuals", fontsize=20)
        plt.savefig(
            "figures/residuals_histogram_p95quantile.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # Histogram of .05 quantile residuals
        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        df_residuals.quantile(0.05, axis=1).hist(bins=30, edgecolor="k")
        plt.xlabel("Residual", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.title("Histogram of min Residuals", fontsize=20)
        plt.savefig(
            "figures/residuals_histogram_p05quantile.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        """
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
        """

        # Group by hour and day and calculate mean residuals
        df_residuals["mean_residuals"] = df_residuals.mean(axis=1)
        grouped = (
            df_residuals.groupby(["hour", "day"])["mean_residuals"]
            .mean()
            .reset_index()
        )
        grouped = grouped.pivot(
            index="hour", columns="day", values="mean_residuals"
        )

        plt.close("all")
        # plt.figure(figsize=FIGSIZE)
        sns.heatmap(grouped, cmap="coolwarm", annot=True, fmt=".2f")
        plt.title("Mean Residuals by Hour and Day of Week", fontsize=20)
        plt.xlabel("Day of Week", fontsize=15)
        plt.ylabel("Hour of Day", fontsize=15)
        plt.savefig(
            "figures/residuals_heatmap.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
