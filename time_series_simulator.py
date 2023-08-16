from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common_constants import CV_STRATEGY, INITIAL_TRAIN_LENGTH, TEST_LENGTH
from time_series_preprocessor import TimeSeriesPreprocessor
from time_series_trainer import TimeSeriesTrainer


class TimeSeriesSimulator:
    """
    Simulates a production environment where new data comes in steps.

    Attributes
    ----------
    STEP_LENGTH : int
        Number of steps to take between each iteration in the walk-forward validation.
    """

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

    def simulate_production(
        self,
        initial_size: int = INITIAL_TRAIN_LENGTH,
        test_length: int = TEST_LENGTH,
        steps: int = 4,
        cv_strategy: str = CV_STRATEGY,
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
            df = self._merge_y_and_weather_data()

            forecaster = TimeSeriesTrainer(self.y_data, self.weather_data)
            df_results = forecaster.forecast(
                df=df,
                target_variable=list(self.y_data)[0],  # type: ignore
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
