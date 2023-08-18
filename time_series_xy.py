from typing import Tuple, Union

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
    """

    @staticmethod
    def create_future_features(
        df: pd.DataFrame,
        target_variables: list[str],
        fhs: Union[int, list[int]] = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        """
        Creates future features for a given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTime index.
        target_variables : list[str]
            List of target variables.
        fhs : Union[int, list[int]]
            Forecast horizons.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the future features.
        """
        # Ensure that index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex")

        # Ensure that target_variables are in df
        for target_variable in target_variables:
            if target_variable not in df.columns:
                raise ValueError(
                    f"target_variable ({target_variable}) not in df.columns"
                )

        # If fh is an int, convert it to a list of ints
        if isinstance(fhs, int):
            fhs = [fhs] * len(target_variables)
        # If fh is a list, ensure that it has the same length as target_variables
        elif isinstance(fhs, list) and len(fhs) != len(target_variables):
            raise ValueError(
                "If fh is a list, it must have the same length as target_variables"
            )

        # Create future features
        future_columns = {}
        for target_variable, fh in zip(target_variables, fhs):
            for i in range(1, fh + 1):
                future_columns[f"{target_variable}_fh_{i}"] = df[
                    target_variable
                ].shift(-i)

        future_df = pd.DataFrame(future_columns)

        return future_df

    @staticmethod
    def df_to_X_y(
        df: pd.DataFrame,
        target_variable: str,
        fh: int = FORECAST_HORIZON,
        dropna: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates regression DataFrame from the given input y and weather data. Optionally add autoregressive features from y or weather data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with a DateTime index, containing both y and optional weather data.
        target_variable: str
            Name of the target (y) variable.
        fh: int
            Forecast horizon.
        dropna: bool
            Whether to drop rows with NaNs.

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
        # Create X
        X = df.drop(columns=target_variable)
        # Ensure that X is a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Create y
        y = TimeSeriesXy.create_future_features(df, [target_variable], fh)

        if dropna:
            # Drop rows with NaNs
            y = y.dropna()
            X = X.loc[y.index]

        return X, y
