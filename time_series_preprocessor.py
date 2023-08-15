from typing import Optional, Tuple, Union

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
        y_file : str
            Path of the CSV file containing y data.
        weather_data_file : str
            Path of the CSV file containing weather data.
        """
        self.y_data = y_data
        self.weather_data = weather_data

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

    def align_timestamps(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
        # Sort indices
        self.y_data.sort_index(inplace=True)  # type: ignore
        # Handle duplicate indices
        self.y_data = TimeSeriesPreprocessor._handle_duplicate_indices(
            self.y_data  # type: ignore
        )
        if self.weather_data is None:
            return self.y_data, None  # type: ignore
        else:
            self.weather_data.sort_index(inplace=True)  # type: ignore
            self.weather_data = TimeSeriesPreprocessor._handle_duplicate_indices(
                self.weather_data  # type: ignore
            )
            common_index = self.y_data.index.intersection(
                self.weather_data.index  # type: ignore
            )
            self.y_data = self.y_data.reindex(common_index, method="ffill")  # type: ignore
            self.weather_data = self.weather_data.reindex(  # type: ignore
                common_index, method="ffill"
            )
            return self.y_data, self.weather_data
