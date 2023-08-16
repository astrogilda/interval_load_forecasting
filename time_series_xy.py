from math import nan
from typing import Tuple

import pandas as pd

from common_constants import (
    AR_FROM_WEATHER_DATA,
    AR_FROM_Y,
    FORECAST_HORIZON,
    LAGS,
    MAX_LAGS,
)
from time_series_featurizer import TimeSeriesFeaturizer


class TimeSeriesXy:
    """
    Class to create X and y for time series regression.

    Attributes
    ----------
    AR_FROM_Y : bool
        Whether to add autoregressive features from y.
    AR_FROM_WEATHER_DATA : bool
        Whether to add autoregressive features from weather data.
    LAGS : int
        Number of lags to use for autoregressive features.
    MAX_LAGS : int
        Maximum number of lags to use for autoregressive features.
    FORECAST_HORIZON : int
        Number of steps to forecast.
    """

    @staticmethod
    def _create_regression_data(
        df: pd.DataFrame,
        target_variable: str,
        fh: int = FORECAST_HORIZON,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with a DateTime index, containing both y and optional weather data.
        target_variable: str
            Name of the target (y) variable.
        fh : int
            Forecast horizon.

        Returns
        -------
        X : pd.DataFrame
            DataFrame ready for regression, with optional weather data and (if requested) autoregressive features.
        y : pd.DataFrame
            DataFrame with the target variable.
        """
        # Create X and y
        X = df.drop(columns=target_variable)
        # Ensure that X is a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Add future values of y
        y_columns = {}
        for i in range(1, fh + 1):
            y_columns[f"y_fh_{i}"] = df[target_variable].shift(i)

        y = pd.DataFrame(y_columns)

        # Drop rows with NaNs
        y = y.dropna()
        X = X.loc[y.index]

        return X, y

    @staticmethod
    def df_to_X_y(
        df: pd.DataFrame, target_variable: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with a DateTime index, containing both y and optional weather data.
        target_variable: str
            Name of the target (y) variable.

        Returns
        -------
        X : pd.DataFrame
            DataFrame ready for regression, with optional weather data and (if requested) autoregressive features.
        y : pd.Series
            Series with the target variable.
        """
        # Create features
        df, _ = TimeSeriesFeaturizer.create_features(
            df,
            target_variable,
            ar_from_y=AR_FROM_Y,
            ar_from_weather_data=AR_FROM_WEATHER_DATA,
            lags=LAGS,
            max_lags=MAX_LAGS,
            use_pacf=False,  # for multi-step forecasting, use_pacf must be False
        )
        # Create X and y
        X, y = TimeSeriesXy._create_regression_data(
            df, target_variable, FORECAST_HORIZON
        )
        return X, y
