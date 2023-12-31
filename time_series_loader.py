from pathlib import Path
from typing import Optional

import pandas as pd

from common_constants import TARGET_VARIABLE, WEATHER_DATA_FILE, Y_FILE


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
        self,
        y_file: str = Y_FILE,
        weather_data_file: Optional[str] = WEATHER_DATA_FILE,
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
            y_file, col_name=TARGET_VARIABLE
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

        # Check if data is a pd.DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # If a column name is specified, and if the dataframe has only one column, rename the column
        if col_name is not None and len(data.columns) == 1:
            data.columns = [col_name]

        return data
