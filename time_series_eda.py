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
    """
    Class for performing Exploratory Data Analysis (EDA) on time series data.
    """

    @staticmethod
    def plot_time_series(
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

        plt.close("all")
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

    @staticmethod
    def plot_seasonal_decomposition(
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
        seasonalities : List[int], optional
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
        if isinstance(seasonalities, int):
            seasonalities = [seasonalities]

        mstl = MSTL(y, periods=seasonalities)
        result = mstl.fit()
        plt.close("all")
        plt.figure(figsize=(16, 12))
        result.plot()
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    @staticmethod
    def plot_acf_pacf(
        y: pd.Series, lags: int, savefig_name: Optional[str] = None
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
        plt.close("all")
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

    @staticmethod
    def plot_distribution(
        y: pd.Series, savefig_name: Optional[str] = None
    ) -> None:
        """
        Plots the distribution of the time series data.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        """
        plt.close("all")
        plt.figure()
        sns.histplot(y, kde=True, edgecolor="black")
        plt.title("Distribution Plot")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    @staticmethod
    def plot_patterns(
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
        plt.close("all")
        plt.figure()
        sns.boxplot(data=df, x=x_variable, y=target_variable)
        plt.title(title)
        plt.xlabel(x_variable)
        plt.ylabel(target_variable)
        if savefig_name is not None:
            plt.savefig(savefig_name)
        else:
            plt.show()

    @staticmethod
    def plot_heatmap(
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
        plt.close("all")
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

    @staticmethod
    def plot_rolling_stats(y: pd.Series, window: Optional[int]) -> None:
        if window is None:
            window = FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY  # 1 day

        y_mean = y.rolling(window=window).mean()
        y_median = y.rolling(window=window).median()
        y_max = y.rolling(window=window).max()
        y_min = y.rolling(window=window).min()

        # Plot basic statistics
        TimeSeriesEDA.plot_time_series(
            [y_mean, y_median, y_max, y_min],
            label=[
                f"Rolling Mean ({window})",
                f"Rolling Median ({window})",
                f"Rolling Max ({window})",
                f"Rolling Min ({window})",
            ],
            color=["blue", "red", "green", "orange"],
            title="Rolling Statistics",
            savefig_name=f"figures/eda/rolling_statistics_{window}.png",
        )

    @staticmethod
    def perform_eda(
        df: pd.DataFrame,
        target_variable: str,
        seasonalities: Optional[list[int]] = None,
        lags: int = FIFTEEN_MINUTES_PER_HOUR
        * HOURS_PER_DAY
        * DAYS_PER_WEEK,  # 1 week
    ) -> None:
        """
        Performs Exploratory Data Analysis (EDA) on the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the time series data.
        target_variable : str
            The target variable to be analyzed.
        seasonalities : list[int], optional
            Seasonalities for seasonal decomposition. Default is None, and is handled in the plot_seasonal_decomposition method.
        lags : int, optional
            Number of lags to consider for ACF and PACF plots. Default is equivalent of 1 week.
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
        TimeSeriesEDA.plot_time_series(
            y,
            title="Load Time Series",
            savefig_name="figures/eda/load_time_series.png",
        )

        # Plot basic statistics
        TimeSeriesEDA.plot_rolling_stats(
            y, window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
        )  # 1 day
        TimeSeriesEDA.plot_rolling_stats(
            y, window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
        )  # 1 week
        TimeSeriesEDA.plot_rolling_stats(
            y, window=FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
        )  # 1 month

        # Plot seasonal decomposition, ACF, PACF
        """
        TimeSeriesEDA.plot_seasonal_decomposition(
            y,
            seasonalities=seasonalities,
            savefig_name="figures/eda/seasonal_decomposition.png",
        )

        TimeSeriesEDA.plot_acf_pacf(
            y, lags, savefig_name="figures/eda/acf_pacf.png"
        )
        """
        TimeSeriesEDA.plot_distribution(
            y, savefig_name="figures/eda/distribution.png"
        )

        # Plot load distributions conditional on hour, day, and month
        for hour in df["hour"].unique():
            subset = df[df["hour"] == hour]
            TimeSeriesEDA.plot_distribution(
                subset[target_variable],
                savefig_name=f"figures/eda/distribution_hour_{hour}.png",
            )
        for day in df["day"].unique():
            subset = df[df["day"] == day]
            TimeSeriesEDA.plot_distribution(
                subset[target_variable],
                savefig_name=f"figures/eda/distribution_day_{day}.png",
            )
        for month in df["month"].unique():
            subset = df[df["month"] == month]
            TimeSeriesEDA.plot_distribution(
                subset[target_variable],
                savefig_name=f"figures/eda/distribution_month_{month}.png",
            )

        # Plot load patterns conditional on hour, day, and month
        TimeSeriesEDA.plot_patterns(
            df,
            target_variable,
            "hour",
            "Hourly Load Patterns",
            savefig_name="figures/eda/hourly_load_patterns.png",
        )
        TimeSeriesEDA.plot_patterns(
            df,
            target_variable,
            "day",
            "Daily Load Patterns",
            savefig_name="figures/eda/daily_load_patterns.png",
        )
        TimeSeriesEDA.plot_patterns(
            df,
            target_variable,
            "month",
            "Monthly Load Patterns",
            savefig_name="figures/eda/monthly_load_patterns.png",
        )
        TimeSeriesEDA.plot_heatmap(
            df,
            target_variable,
            ["day", "hour"],
            "Weekly Load Heatmap",
            savefig_name="figures/eda/weekly_load_heatmap.png",
        )
        TimeSeriesEDA.plot_heatmap(
            df,
            target_variable,
            ["month", "hour"],
            "Annual Load Heatmap",
            savefig_name="figures/eda/annual_load_heatmap.png",
        )
