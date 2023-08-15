from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from statsmodels.tsa.stattools import pacf


class TimeSeriesFeaturizer:
    """
    Prepares time series data for regression.

    Methods
    -------
    _create_calendar_features
        Creates calendar features from a DataFrame with a DateTime index.
    _find_best_lag_pacf
        Finds the best lag for a time series using PACF.
    create_regression_data
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.
    """

    @staticmethod
    def create_calendar_features(
        df: pd.DataFrame, normalize: bool = True
    ) -> pd.DataFrame:
        """
        Creates calendar features from a DataFrame with a DateTime index.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTime index.
        normalize : bool, optional
            If True, normalize the features to [0, 1]. Default is True.

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
        df["month"] = (
            (df.index.month + 1) / 12 if normalize else df.index.month
        )
        df["dayofmonth"] = df.index.day / 31 if normalize else df.index.day
        df["dayofweek"] = (
            (df.index.dayofweek + 1) / 7 if normalize else df.index.dayofweek
        )
        df["hour"] = (df.index.hour + 1) / 24 if normalize else df.index.hour
        df["minute"] = (
            (df.index.minute + 1) / 60 if normalize else df.index.minute
        )

        return df

    @staticmethod
    def find_best_lag_pacf(y: ArrayLike, max_lag: int) -> int:
        """
        Finds the best lag for a time series using PACF.

        Parameters
        ----------
        y : ArrayLike
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
        best_lag = int(np.argmax(lag_pacf[1:])) + 1

        return best_lag

    @staticmethod
    def create_regression_data(
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
        weather_data : pd.DataFrame, optional
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

        if y.empty:
            raise ValueError("Time series data cannot be empty.")

        # Merge y with weather data
        if weather_data is None:
            df = y.copy()
        else:
            df = pd.merge(
                y,
                weather_data,
                left_index=True,
                right_index=True,
                how="inner",
            )

        # Create calendar features
        df = TimeSeriesFeaturizer.create_calendar_features(df)

        # If autoregressive features from y are requested, add them
        y_name = y.columns[0]
        if ar_from_y:
            if use_pacf:
                best_lag_y = TimeSeriesFeaturizer.find_best_lag_pacf(
                    y, max_lags
                )
            else:
                best_lag_y = lags
            for i in range(1, best_lag_y + 1):
                df[f"y_lag_{i}"] = df[y_name].shift(i)

        # If autoregressive features from weather_data are requested, add them
        if weather_data is not None and ar_from_weather_data:
            for column in weather_data.columns:
                if use_pacf:
                    best_lag_column = TimeSeriesFeaturizer.find_best_lag_pacf(
                        df[column], max_lags
                    )
                else:
                    best_lag_column = lags
                for i in range(1, best_lag_column + 1):
                    df[f"{column}_lag_{i}"] = df[column].shift(i)

        return df.dropna()
