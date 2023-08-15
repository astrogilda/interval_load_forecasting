import os
from typing import Optional, Tuple, Union

import pandas as pd


class TimeSeriesPreprocessor:
    """
    Handles data preprocessing for time series forecasting, including loading, handling duplicates, aligning timestamps,
    and creating features.

    Attributes
    ----------
    y_file : str
        Path of the CSV file containing y data.
    weather_data_file : Optional[str]
        Path of the CSV file containing weather data.
    """

    def __init__(
        self, y_file: str, weather_data_file: Optional[str] = None
    ) -> None:
        """
        Constructs all the necessary attributes for the TimeSeriesPreprocessor object.

        Parameters
        ----------
        y_file : str
            Path of the CSV file containing y data.
        weather_data_file : str
            Path of the CSV file containing weather data.
        """
        self.y_data, self.weather_data = self._load_data(
            y_file
        ), self._load_data(weather_data_file)

    @staticmethod
    def _load_data(file_name: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Load data from CSV files and perform sanity checks.

        Parameters
        ----------
        file_name : str
            Path of the CSV file containing the data.

        Returns
        -------
        data : pd.DataFrame
            Dataframe containing the data.
        """
        # If file_name is None, return None
        if file_name is None:
            return None

        # Check if file exists
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} not found")

        # Load data
        data = pd.read_csv(file_name, index_col=0)
        # Rename index to "DateTime"
        data.index.rename("DateTime", inplace=True)
        # Convert the index to datetime type
        data.index = pd.to_datetime(data.index)

        # Check if data is a pd.DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check for missing values and forward fill
        data = data.fillna(method="ffill")

        return data

    @staticmethod
    def _handle_duplicate_indices(
        *dfs: pd.DataFrame,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        """
        Averages rows with duplicate indices in one or more pandas DataFrames.

        Parameters
        ----------
        dfs : tuple of pd.DataFrame
            One or more pandas DataFrames with timestamps as index.

        Returns
        -------
        output_dfs : pd.DataFrame or tuple of pd.DataFrame
            DataFrame(s) with unique indices. If a single DataFrame was passed as input, a single DataFrame is returned.
            If multiple DataFrames were passed, a tuple of DataFrames is returned.
        """
        output_dfs = []
        for df in dfs:
            # Check if df is a pandas DataFrame or Series
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    "All inputs must be pandas DataFrames or Series"
                )

            # Group by index and average rows with duplicate indices
            df = df.groupby(df.index).mean()
            output_dfs.append(df)

        if len(output_dfs) == 1:
            return output_dfs[0]  # return single dataframe
        else:
            return tuple(output_dfs)  # return tuple of dataframes

    @staticmethod
    def _align_timestamps(
        df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns the timestamps of two dataframes, in case they have different frequencies or missing time points.

        Parameters
        ----------
        df1 : pd.DataFrame
            First DataFrame.
        df2 : pd.DataFrame
            Second DataFrame.

        Returns
        -------
        df1_aligned, df2_aligned : tuple of pd.DataFrame
            Aligned dataframes.
        """
        # Ensure the DataFrames are sorted by the index
        df1.sort_index(inplace=True)
        df2.sort_index(inplace=True)

        # Ensure the indices are unique
        df1, df2 = TimeSeriesPreprocessor._handle_duplicate_indices(df1, df2)

        # Reindex the DataFrames to match the union of the two indices and forward fill any missing data
        common_index = df1.index.intersection(df2.index)
        df1_aligned = df1.reindex(common_index, method="ffill")
        df2_aligned = df2.reindex(common_index, method="ffill")

        return df1_aligned, df2_aligned

    @staticmethod
    def _create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates calendar features from a DataFrame with a DateTime index.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTime index.

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
        df["month"] = df.index.month / 12  # normalize to [0, 1]
        df["day"] = df.index.day / 31  # normalize to [0, 1]
        df["dayofweek"] = df.index.dayofweek / 7  # normalize to [0, 1]
        df["hour"] = df.index.hour / 24  # normalize to [0, 1]
        df["minute"] = df.index.minute / 60  # normalize to [0, 1]

        return df

    @staticmethod
    def _create_regression_data(
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
        weather_data : pd.DataFrame
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

        # Merge y with weather data
        if weather_data is None:
            df = y.copy()
        else:
            df = pd.merge(
                y, weather_data, left_index=True, right_index=True, how="inner"
            )

        # Create calendar features
        df = TimeSeriesPreprocessor._create_calendar_features(df)

        # If autoregressive features from y are requested, add them
        if ar_from_y:
            if use_pacf:
                best_lag_y = TimeSeriesRegression._find_best_lag_pacf(
                    y, max_lags
                )
            else:
                best_lag_y = lags
            for i in range(1, best_lag_y + 1):
                df[f"y_lag_{i}"] = df[y.name].shift(i)

        # If autoregressive features from weather_data are requested, add them
        if weather_data is not None and ar_from_weather_data:
            for column in weather_data.columns:
                if use_pacf:
                    best_lag_column = TimeSeriesRegression._find_best_lag_pacf(
                        df[column], max_lags
                    )
                else:
                    best_lag_column = lags
                for i in range(1, best_lag_column + 1):
                    df[f"{column}_lag_{i}"] = df[column].shift(i)

        return df.dropna()
