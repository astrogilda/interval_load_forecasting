from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesPreprocessor:
    """
    Handles data preprocessing for time series forecasting, including loading, handling duplicates, aligning timestamps, and creating features.

    Methods
    -------
    _handle_duplicate_indices(*dfs)
        Averages rows with duplicate indices in one or more pandas DataFrames.
    align_timestamps(df1, df2)
        Aligns the timestamps of two dataframes, in case they have different frequencies or missing time points.
    """

    def __init__(
        self, y_data: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Constructs all the necessary attributes for the TimeSeriesPreprocessor object.

        Parameters
        ----------
        y_data : pd.DataFrame
            DataFrame containing the target variable.
        weather_data : pd.DataFrame, optional
            DataFrame containing the weather data. The default is None.
        """
        self.y_data = y_data
        self.weather_data = weather_data

    @staticmethod
    def _handle_missing_indices(
        *dfs: Union[pd.DataFrame, pd.Series], freq: Optional[str] = None
    ) -> Union[
        Union[pd.DataFrame, pd.Series], Tuple[Union[pd.DataFrame, pd.Series]]
    ]:
        """
        Fills missing indices, missing values, and infinite values in one or more pandas DataFrames or Series.

        Parameters
        ----------
        dfs : tuple of Union[pd.DataFrame, pd.Series]
            One or more pandas DataFrames or Series with timestamps as index.
        freq : str, optional
            Frequency of the time series data. If None, the frequency will be inferred from the data (default is None).

        Returns
        -------
        output_dfs : Union[pd.DataFrame, pd.Series] or tuple of Union[pd.DataFrame, pd.Series]
            DataFrame(s) or Series with missing indices, values, and infinite values filled. If a single DataFrame or Series was passed as input,
            a single DataFrame or Series is returned. If multiple DataFrames or Series were passed, a tuple of DataFrames or Series is returned.
        """
        output_dfs = []
        for df in dfs:
            # Check if df is a pandas DataFrame or Series
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    "All inputs must be pandas DataFrames or Series"
                )

            # Check if df has a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("All inputs must have a DatetimeIndex")

            # If frequency is not provided, infer it from the DataFrame or Series
            if freq is None:
                freq = pd.infer_freq(df.index)

                if freq is None:
                    raise ValueError(
                        "Frequency could not be inferred from the data. Please provide the 'freq' parameter."
                    )

            # Create a complete date range for the given frequency
            full_range = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq=freq
            )

            # Reindex the DataFrame or Series to ensure all expected indices are present
            df = df.reindex(full_range)

            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Forward fill missing values (including NaN and previously infinite values)
            df = df.fillna(method="ffill")

            output_dfs.append(df)

        if len(output_dfs) == 1:
            return output_dfs[0]
        else:
            return tuple(output_dfs)

    @staticmethod
    def _handle_duplicate_indices(
        *dfs: Union[pd.DataFrame, pd.Series]
    ) -> Union[
        Union[pd.DataFrame, pd.Series], Tuple[Union[pd.DataFrame, pd.Series]]
    ]:
        """
        Averages rows with duplicate indices in one or more pandas DataFrames or Series.

        Parameters
        ----------
        dfs : tuple of Union[pd.DataFrame, pd.Series]
            One or more pandas DataFrames or Series with timestamps as index.

        Returns
        -------
        output_dfs : Union[pd.DataFrame, pd.Series] or tuple of Union[pd.DataFrame, pd.Series]
            DataFrame(s) or Series with unique indices. If a single DataFrame or Series was passed as input, a single DataFrame or Series is returned.
            If multiple DataFrames or Series were passed, a tuple of DataFrames or Series is returned.
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
            return output_dfs[0]
        else:
            return tuple(output_dfs)

    @staticmethod
    def _align_timestamps(
        *dfs: Union[pd.DataFrame, pd.Series]
    ) -> Union[
        Union[pd.DataFrame, pd.Series], Tuple[Union[pd.DataFrame, pd.Series]]
    ]:
        """
        Aligns the timestamps of multiple dataframes or series, in case they have different frequencies or missing time points.

        Parameters
        ----------
        dfs : tuple of Union[pd.DataFrame, pd.Series]
            One or more pandas DataFrames or Series with timestamps as index.

        Returns
        -------
        output_dfs : Union[pd.DataFrame, pd.Series] or tuple of Union[pd.DataFrame, pd.Series]
            Aligned DataFrame(s) or Series. If a single DataFrame or Series was passed as input, a single DataFrame or Series is returned.
            If multiple DataFrames or Series were passed, a tuple of DataFrames or Series is returned.
        """
        # Find common index across all DataFrames or Series
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)

        # Reindex and forward fill missing values
        output_dfs = [df.reindex(common_index, method="ffill") for df in dfs]

        if len(output_dfs) == 1:
            return output_dfs[0]
        else:
            return tuple(output_dfs)

    def merge_y_and_weather_data(self) -> pd.DataFrame:
        """
        Merges y_data and weather_data.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame.
        """
        self._align_timestamps()
        if self.weather_data is None:
            return self.y_data
        else:
            return pd.merge(
                self.y_data,
                self.weather_data,
                left_index=True,
                right_index=True,
                how="inner",
            )
