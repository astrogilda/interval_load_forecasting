from typing import Tuple

import pandas as pd

from time_constants import (
    DAYS_PER_WEEK,
    FIFTEEN_MINUTES_PER_HOUR,
    HOURS_PER_DAY,
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

    AR_FROM_Y = True  # Autoregressive features from y
    AR_FROM_WEATHER_DATA = False  # Autoregressive features from weather data
    LAGS = (
        FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
    )  # Number of lags to use for autoregressive features; 1 week
    MAX_LAGS = (
        FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
    )  # Maximum number of lags to use for autoregressive features; 1 week
    FORECAST_HORIZON = (
        FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
    )  # Number of steps to forecast; 1 day

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

        y_columns = {}
        for i in range(fh):
            y_columns[f"y_fh_{i+1}"] = df[target_variable].shift(i)

        y = pd.DataFrame(y_columns)

        # Drop NaN values from y and corresponding rows from X
        nan_rows = y.isna()
        y = y.dropna()
        X = X.loc[~nan_rows]

        return X, y

    def df_to_X_y(
        self, df: pd.DataFrame, target_variable: str
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
            ar_from_y=self.AR_FROM_Y,
            ar_from_weather_data=self.AR_FROM_WEATHER_DATA,
            lags=self.LAGS,
            max_lags=self.MAX_LAGS,
            use_pacf=False,  # for multi-step forecasting, use_pacf must be False
        )
        # Create X and y
        X, y = TimeSeriesXy._create_regression_data(
            df, target_variable, self.FORECAST_HORIZON
        )
        return X, y
