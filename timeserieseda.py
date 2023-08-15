from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import MSTL

# Default figure size for plots
# FIGSIZE = (16, 8)

# Define the settings for high-quality plots
plot_settings = {
    "figure.figsize": (14, 8),  # Figure size
    "font.size": 12,  # Font size
    "font.family": "serif",  # Font family
    "lines.linewidth": 3,  # Line width
    "axes.labelsize": 16,  # Label size
    "axes.titlesize": 18,  # Title size
    "xtick.labelsize": 14,  # X-tick label size
    "ytick.labelsize": 14,  # Y-tick label size
    "legend.fontsize": 14,  # Legend font size
    "axes.grid": True,  # Show grid
    "savefig.dpi": 300,  # DPI for saving figures
    "savefig.format": "png",  # Format for saving figures
    "savefig.bbox": "tight",  # Bounding box for saving figures
    "savefig.pad_inches": 0.1,  # Padding for saving figures
}


plt.style.use("dark_background")

# Apply the settings globally
plt.rcParams.update(plot_settings)


class TimeSeriesEDA:
    def plot_time_series(self, y: pd.Series) -> None:
        """
        Plots the time series data.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        """
        plt.figure()  # figsize=FIGSIZE)
        plt.plot(y)
        plt.title("Time Series Plot")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

    def plot_seasonal_decomposition(
        self,
        y: pd.Series,
        seasonalities: Optional[List[int]] = None,
    ) -> None:
        """
        Plots the seasonal decomposition of the time series data using STL.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        seasonalities : Union[int, List[int]], optional
            Seasonalities to consider for decomposition. Default is [96, 96*7, 96*30].
        """
        if seasonalities is None:
            seasonalities = [
                96,
                96 * 7,
                96 * 30,
            ]  # 96 15-minute intervals in a day, 96*7 in a week
        mstl = MSTL(y, periods=seasonalities)
        result = mstl.fit()
        plt.close("all")
        result.plot()

    def plot_acf_pacf(self, y: pd.Series, lags: int) -> None:
        """
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

        Parameters
        ----------
        y : pd.Series
            Time series data.
        lags : int
            Number of lags to consider.
        """
        plt.figure()
        plt.subplot(211)
        autocorrelation_plot(y)
        plt.title("Autocorrelation Function (ACF)")

        plt.subplot(212)
        plot_pacf(y, lags=lags)
        plt.title("Partial Autocorrelation Function (PACF)")
        plt.show()

    def plot_distribution(self, y: pd.Series) -> None:
        """
        Plots the distribution of the time series data.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        """
        plt.figure()
        sns.histplot(y, kde=True, edgecolor="black")
        plt.title("Distribution Plot")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    def plot_patterns(
        self,
        df: pd.DataFrame,
        target_variable: str,
        x_variable: str,
        title: str,
    ) -> None:
        """
        Plots the patterns of the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the time series data.
        target_variable : str
            The target variable to be plotted.
        x_variable : str
            The variable to be plotted on the x-axis.
        title : str
            The title of the plot.
        """
        plt.figure()
        sns.boxplot(data=df, x=x_variable, y=target_variable)
        plt.title(title)
        plt.show()

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        target_variable: str,
        groupby_vars: list,
        title: str,
    ) -> None:
        """
        Plots a heatmap of the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the time series data.
        target_variable : str
            The target variable to be plotted.
        groupby_vars : list
            List of variables to group by for the heatmap.
        title : str
            The title of the heatmap.
        """
        load_data = df.groupby(groupby_vars)[target_variable].mean().unstack()
        plt.figure()
        sns.heatmap(
            load_data,
            cmap="coolwarm",
            annot=True,
            fmt=".0f",
            linewidths=1,
            linecolor="black",
        )
        plt.title(title)
        plt.show()

    def perform_eda(
        self,
        df: pd.DataFrame,
        target_variable: str,
        freq: int = 24,
        lags: int = 40,
    ) -> None:
        """
        Performs Exploratory Data Analysis (EDA) on the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the time series data.
        target_variable : str
            The target variable to be analyzed.
        freq : int, optional
            Frequency of the seasonality for seasonal decomposition. Default is 24.
        lags : int, optional
            Number of lags to consider for ACF and PACF plots. Default is 40.
        """
        # Ensure that we do not modify the original dataframe
        df = df.copy()

        # Ensure that the index is a datetime object
        df.index = pd.to_datetime(df.index)

        # Extract necessary information
        df["hour"] = df.index.hour.values
        df["day"] = df.index.dayofweek
        df["month"] = df.index.month

        # Call the plotting methods
        y = df[target_variable]
        self.plot_time_series(y)
        # self.plot_seasonal_decomposition(y, freq)
        self.plot_acf_pacf(y, lags)
        self.plot_distribution(y)
        self.plot_patterns(df, target_variable, "hour", "Hourly Load Patterns")
        self.plot_patterns(df, target_variable, "day", "Daily Load Patterns")
        self.plot_patterns(
            df, target_variable, "month", "Monthly Load Patterns"
        )
        self.plot_heatmap(
            df, target_variable, ["day", "hour"], "Weekly Load Heatmap"
        )
        self.plot_heatmap(
            df, target_variable, ["month", "hour"], "Annual Load Heatmap"
        )
