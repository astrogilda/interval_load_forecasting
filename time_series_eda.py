from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import MSTL

from time_constants import (
    DAYS_PER_MONTH,
    DAYS_PER_WEEK,
    FIFTEEN_MINUTES_PER_HOUR,
    HOURS_PER_DAY,
)

# Define the settings for high-quality plots
plot_settings = {
    "figure.figsize": (16, 8),  # Figure size
    "font.size": 15,  # Font size
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
    def plot_time_series(
        self,
        y: Union[pd.Series, list[pd.Series]],
        title: str,
        label: Union[str, list[str]] = "",
        color: Union[str, list[str]] = "blue",
        savefig_name: Optional[str] = None,
    ) -> None:
        """
        Plots the time series data.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        """
        # If y or label or color is list, ensure that they are all lists of the same length
        if (
            isinstance(y, list)
            and isinstance(label, list)
            and isinstance(color, list)
        ):
            if len(y) != len(label) or len(y) != len(color):
                raise ValueError(
                    "If y, label, and color are lists, they must all have the same length"
                )
        # If y or label or color is not a list, convert it to a list
        elif (
            not isinstance(y, list)
            and not isinstance(label, list)
            and not isinstance(color, list)
        ):
            y = [y]
            label = [label]
            color = [color]
        else:
            raise TypeError(
                "If y, label, and color are lists, they must all have the same length"
            )

        plt.figure()
        for y_, label_, color_ in zip(y, label, color):
            plt.plot(y_, label=label_, color=color_)
        plt.legend(loc="best", fontsize=12)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Load")
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    def plot_seasonal_decomposition(
        self,
        y: pd.Series,
        seasonalities: Optional[List[int]] = None,
        savefig_name: Optional[str] = None,
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
                FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY,  # 1 day
                FIFTEEN_MINUTES_PER_HOUR
                * HOURS_PER_DAY
                * DAYS_PER_WEEK,  # 1 week
                FIFTEEN_MINUTES_PER_HOUR
                * HOURS_PER_DAY
                * DAYS_PER_MONTH,  # 1 month
            ]
        mstl = MSTL(y, periods=seasonalities)
        result = mstl.fit()
        plt.close("all")
        result.plot()
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    def plot_acf_pacf(
        self, y: pd.Series, lags: int, savefig_name: Optional[str] = None
    ) -> None:
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

        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    def plot_distribution(
        self, y: pd.Series, savefig_name: Optional[str] = None
    ) -> None:
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
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    def plot_patterns(
        self,
        df: pd.DataFrame,
        target_variable: str,
        x_variable: str,
        title: str,
        savefig_name: Optional[str] = None,
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
        plt.xlabel(x_variable)
        plt.ylabel(target_variable)
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        target_variable: str,
        groupby_vars: list,
        title: str,
        savefig_name: Optional[str] = None,
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
        plt.xlabel(groupby_vars[1])
        plt.ylabel(groupby_vars[0])
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
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
        self.plot_time_series(
            y, title="Load Time Series", savefig_name="load_time_series.png"
        )

        # Calculate basic statistics
        y_mean = y.rolling(
            window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
        ).mean()
        y_median = y.rolling(
            window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
        ).median()
        y_max = y.rolling(
            window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
        ).max()
        y_min = y.rolling(
            window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
        ).min()

        # Plot basic statistics
        self.plot_time_series(
            [y_mean, y_median, y_max, y_min],
            label=[
                "Rolling Mean (24h)",
                "Rolling Median (24h)",
                "Rolling Max (24h)",
                "Rolling Min (24h)",
            ],
            title="Rolling Statistics",
            savefig_name="rolling_statistics.png",
        )

        # self.plot_seasonal_decomposition(y, freq)
        self.plot_acf_pacf(y, lags, savefig_name="acf_pacf.png")
        self.plot_distribution(y, savefig_name="distribution.png")
        self.plot_patterns(
            df,
            target_variable,
            "hour",
            "Hourly Load Patterns",
            savefig_name="hourly_load_patterns.png",
        )
        self.plot_patterns(
            df,
            target_variable,
            "day",
            "Daily Load Patterns",
            savefig_name="daily_load_patterns.png",
        )
        self.plot_patterns(
            df,
            target_variable,
            "month",
            "Monthly Load Patterns",
            savefig_name="monthly_load_patterns.png",
        )
        self.plot_heatmap(
            df,
            target_variable,
            ["day", "hour"],
            "Weekly Load Heatmap",
            savefig_name="weekly_load_heatmap.png",
        )
        self.plot_heatmap(
            df,
            target_variable,
            ["month", "hour"],
            "Annual Load Heatmap",
            savefig_name="annual_load_heatmap.png",
        )
