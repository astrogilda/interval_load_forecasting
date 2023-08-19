import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.stattools import pacf

from common_constants import (
    AR_FROM_WEATHER_DATA,
    AR_FROM_Y,
    LAGS,
    MAX_LAGS,
    TARGET_VARIABLE,
)
from time_constants import MONTHS_PER_YEAR
from time_series_preprocessor import TimeSeriesPreprocessor


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
    def _cyclical_encoding(
        df: pd.DataFrame, columns: list[str], max_vals: list
    ) -> pd.DataFrame:
        """
        Encodes a cyclical feature using sine and cosine transformations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTime index.
        columns : list[str]
            List of column names to encode.
        max_vals : list
            List of maximum values for each column.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the encoded features.
        """
        # Ensure that index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Ensure that columns and max_vals have the same length
        if len(columns) != len(max_vals):
            raise ValueError("columns and max_vals must have the same length")

        # Encode the cyclical features
        for column, max_val in zip(columns, max_vals):
            df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / max_val)
            df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / max_val)
            df.drop(columns=column, inplace=True)

        return df

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
        df["month"] = df.index.month
        df["dayofmonth"] = df.index.day
        df["dayofweek"] = df.index.dayofweek
        df["hourofday"] = df.index.hour
        df["minuteofhour"] = df.index.minute

        if normalize:
            TimeSeriesFeaturizer._cyclical_encoding(
                df,
                [
                    "month",
                    "dayofmonth",
                    "dayofweek",
                    "hourofday",
                    "minuteofhour",
                ],
                [
                    MONTHS_PER_YEAR,
                    df.index.days_in_month,
                    7 - 1,
                    24 - 1,
                    60 - 1,
                ],
            )

        return df

    @staticmethod
    def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a binary feature indicating US federal holidays to a DataFrame with a DateTime index.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTime index.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with a new binary feature 'is_holiday', where 1 indicates a US federal holiday and 0 indicates otherwise.
        """
        # Ensure that index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Get US federal holidays
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df.index.min(), end=df.index.max())

        # Create a binary feature indicating whether the date is a US federal holiday
        df["is_holiday"] = np.isin(df.index, holidays).astype(int)

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
    def stats_features(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Creates summary statistics features from a DataFrame with features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with features summarized.
        """
        # Ensure that index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Ensure that df has at least 8 columns
        if len(df.columns) <= 7:
            return df

        # Summarize features
        df_stats = pd.DataFrame(index=df.index)
        df_stats[f"{col_name}_mean"] = df.mean(axis=1)
        df_stats[f"{col_name}_std"] = df.std(axis=1)
        df_stats[f"{col_name}_min"] = df.min(axis=1)
        df_stats[f"{col_name}_max"] = df.max(axis=1)
        df_stats[f"{col_name}_median"] = df.median(axis=1)
        df_stats[f"{col_name}_skew"] = df.skew(axis=1)
        df_stats[f"{col_name}_kurtosis"] = df.kurtosis(axis=1)

        return df_stats

    @staticmethod
    def create_ar_features(
        series: pd.Series, lags: int, use_pacf: bool = False, max_lags: int = 3
    ) -> pd.DataFrame:
        """
        Creates autoregressive features for a given time series.

        Parameters
        ----------
        series : pd.Series
            Time series data.
        lags : int
            Number of autoregressive features to add.
        use_pacf : bool
            If True, use PACF to find the best lag.
        max_lags : int
            Maximum number of lags possible.

        Returns
        -------
        ar_features : pd.DataFrame
            DataFrame containing the autoregressive features.
        """
        if use_pacf:
            best_lag = TimeSeriesFeaturizer.find_best_lag_pacf(
                series.dropna(), max_lags
            )
        else:
            best_lag = lags

        ar_features = pd.DataFrame(
            {
                f"{series.name}_lag_{i}": series.shift(i)
                for i in range(best_lag)
            }
        )
        ar_features.index = series.index

        # Drop the newly created NaNs at the beginning, but retain existing NaNs in the original series
        ar_features = ar_features.iloc[best_lag - 1 :]

        # Create summary statistics features
        ar_features = TimeSeriesFeaturizer.stats_features(
            ar_features, str(series.name)
        )

        return ar_features

    @staticmethod
    def remove_zero_variability_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes features with zero variability from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with features with zero variability removed.
        """
        # Ensure that index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Remove features with zero variability
        df = df.loc[:, df.std() != 0]

        return df

    @staticmethod
    def create_features(
        df: pd.DataFrame,
        target_variable: str = TARGET_VARIABLE,
        ar_from_y: bool = AR_FROM_Y,
        ar_from_weather_data: bool = AR_FROM_WEATHER_DATA,
        lags: int = LAGS,
        max_lags: int = MAX_LAGS,
        use_pacf: bool = False,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with a DateTime index, containing both y and optional weather data.
        target_variable: str
            Name of the target (y) variable.
        ar_from_y : bool
            If True, add autoregressive features from y.
        ar_from_weather_data : bool
            If True, add autoregressive features from weather data.
        lags : int
            Number of autoregressive features to add.
        max_lags : int
            Maximum number of lags possible; this is a function of step_length in the cross-validator.
        use_pacf : bool
            If True, use PACF to find the best lag.
        drop_na : bool
            If True, drop NaNs.

        Returns
        -------
        df : pd.DataFrame
            DataFrame ready for regression, with weather data and (if requested) autoregressive features.
        """
        # Ensure that target_variable is in df
        if target_variable not in df.columns:
            raise ValueError(
                f"target_variable ({target_variable}) not in df.columns"
            )

        # Ensure that the index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Don't modify the original DataFrame
        df = df.copy()

        y = df[target_variable]
        weather_data = df.drop(columns=target_variable)

        # Create calendar features
        df = TimeSeriesFeaturizer.create_calendar_features(df)

        # Create holiday features
        df = TimeSeriesFeaturizer.create_holiday_features(df)

        # If autoregressive features from y are requested, add them
        if ar_from_y:
            y_ar_features = TimeSeriesFeaturizer.create_ar_features(
                y, lags, use_pacf, max_lags
            )
            df, y_ar_features = TimeSeriesPreprocessor._align_timestamps(df, y_ar_features)  # type: ignore
            df = pd.merge(
                df,  # type: ignore
                y_ar_features,  # type: ignore
                left_index=True,
                right_index=True,
                how="inner",
            )

        # If autoregressive features from weather_data are requested, add them
        if list(weather_data) and ar_from_weather_data:
            for column in weather_data.columns:
                column_ar_features = TimeSeriesFeaturizer.create_ar_features(
                    weather_data[column], lags, use_pacf, max_lags
                )
                df, column_ar_features = TimeSeriesPreprocessor._align_timestamps(df, column_ar_features)  # type: ignore
                df = pd.merge(
                    df,  # type: ignore
                    column_ar_features,  # type: ignore
                    left_index=True,
                    right_index=True,
                    how="inner",
                )

        # Remove features with zero variability
        # df = TimeSeriesFeaturizer.remove_zero_variability_features(df)

        # Drop NaNs
        if drop_na:
            df.dropna(inplace=True)

        return df
