from pathlib import Path
from typing import Optional

import pandas as pd


class TimeSeriesLoader:
    """
    Handles data loading for time series forecasting.

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
        Constructs all the necessary attributes for the TimeSeriesLoader object.

        Parameters
        ----------
        y_file : str
            Path of the CSV file containing y data.
        weather_data_file : str
            Path of the CSV file containing weather data.
        """
        self.y_data, self.weather_data = self.load_data(
            y_file, col_name="Load"
        ), self.load_data(weather_data_file)

    @staticmethod
    def load_data(
        file_name: Optional[str], col_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
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
        if not Path(file_name).is_file():
            raise FileNotFoundError(f"File {file_name} not found")

        # Load data
        data = pd.read_csv(file_name, index_col=0)
        # Rename index to "DateTime"
        data.index.rename("DateTime", inplace=True)
        # Convert the index to datetime type
        data.index = pd.to_datetime(data.index)

        # Compute the differences between consecutive timestamps
        deltas = data.index.to_series().diff().value_counts().index
        # Take the most common difference as the inferred frequency
        inferred_freq = deltas[0] if len(deltas) > 0 else None
        # Create a new index with the inferred frequency
        if inferred_freq:
            data = data.asfreq(inferred_freq)

        # Check if data is a pd.DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check for missing values and forward fill
        data = data.fillna(method="ffill")

        # If a column name is specified, and if the dataframe has only one column, rename the column
        if col_name is not None and len(data.columns) == 1:
            data.columns = [col_name]

        return data
