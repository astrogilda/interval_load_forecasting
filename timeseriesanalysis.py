from typing import Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FIGSIZE = (16, 8)


class TimeSeriesAnalysis:
    """
    Class for time series analysis, including exploratory data analysis (EDA), error analysis, and visualization.

    Attributes
    ----------
    figsize : tuple
        Figure size for plots.
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 8)) -> None:
        """
        Constructs all the necessary attributes for the TimeSeriesAnalysis object.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size for plots, by default (16, 8).
        """
        self.figsize = figsize

    def plot_series(
        self, series: pd.Series, title: str, xlabel: str, ylabel: str
    ) -> None:
        """
        Plot a time series.

        Parameters
        ----------
        series : pd.Series
            Time series data.
        title : str
            Plot title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        """
        plt.figure(figsize=self.figsize)
        series.plot()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_residuals(self, residuals: pd.Series) -> None:
        """
        Plot residuals.

        Parameters
        ----------
        residuals : pd.Series
            Residuals data.
        """
        plt.figure(figsize=self.figsize)
        residuals.plot(kind="line", color="red")
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.title("Residuals Over Time")
        plt.xlabel("Datetime")
        plt.ylabel("Residual")
        plt.show()

    def plot_actual_vs_predicted(
        self, actual: pd.Series, predicted: pd.Series
    ) -> None:
        """
        Scatter plot of actual vs. predicted values.

        Parameters
        ----------
        actual : pd.Series
            Actual values.
        predicted : pd.Series
            Predicted values.
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot(
            [actual.min(), actual.max()],
            [actual.min(), actual.max()],
            "k--",
            label="Perfect Fit",
        )
        plt.xlabel("True Load")
        plt.ylabel("Predicted Load")
        plt.legend()
        plt.title("Actual vs. Predicted Load")
        plt.show()

    def plot_residuals_heatmap(self, residuals: pd.Series) -> None:
        """
        Heatmap of residuals by hour and day of the week.

        Parameters
        ----------
        residuals : pd.Series
            Residuals data.
        """
        residuals.index = pd.to_datetime(residuals.index)
        residuals_df = pd.DataFrame(
            {
                "hour": residuals.index.hour,
                "day": residuals.index.dayofweek,
                "residuals": residuals,
            }
        )
        grouped = (
            residuals_df.groupby(["hour", "day"])["residuals"]
            .mean()
            .reset_index()
            .pivot(index="hour", columns="day", values="residuals")
        )
        plt.figure(figsize=self.figsize)
        sns.heatmap(grouped, cmap="coolwarm", annot=True, fmt=".2f")
        plt.title("Mean Residuals by Hour and Day of Week")
        plt.xlabel("Day of Week")
        plt.ylabel("Hour of Day")
        plt.show()
