from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import STL

# Default figure size for plots
FIGSIZE = (16, 8)


class TimeSeriesEDA:
    def plot_time_series(self, y: pd.Series) -> None:
        """
        Plots the time series data.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        """
        plt.figure(figsize=FIGSIZE)
        plt.plot(y)
        plt.title("Time Series Plot")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

    def plot_seasonal_decomposition(
        self,
        y: pd.Series,
        seasonal: Optional[int] = None,
    ) -> None:
        """
        Plots the seasonal decomposition of the time series data using STL.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        seasonal : int, optional
            Frequency of the seasonality. Default is None.
        """
        stl = STL(y) if seasonal is None else STL(y, seasonal=seasonal)
        result = stl.fit()
        result.plot()
        plt.show()

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
        plt.figure(figsize=FIGSIZE)
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
        plt.figure(figsize=FIGSIZE)
        sns.histplot(y, kde=True)
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
        plt.figure(figsize=FIGSIZE)
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
        plt.figure(figsize=FIGSIZE)
        sns.heatmap(load_data, cmap="coolwarm")
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
        #self.plot_seasonal_decomposition(y, freq)
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
