import numpy as np
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)

from time_series_eda import TimeSeriesEDA
from time_series_featurizer import TimeSeriesFeaturizer
from time_series_loader import TimeSeriesLoader
from time_series_preprocessor import TimeSeriesPreprocessor
from time_series_simulator import TimeSeriesSimulator
from time_series_trainer import TimeSeriesTrainer
from time_series_xy import TimeSeriesXy

# Load data
data_loader = TimeSeriesLoader()
y_data, weather_data = data_loader.y_data, data_loader.weather_data

# Perform EDA
eda = TimeSeriesEDA
eda.perform_eda(y_data, target_variable="Load")  # type: ignore

# Preprocess data
preprocessor = TimeSeriesPreprocessor(y_data, weather_data)
df = preprocessor.merge_y_and_weather_data(use_union=False)

# Simulate production
simulator = TimeSeriesSimulator(df)
simulator.simulate_production()

# Plot simulated production
TimeSeriesSimulator.plot_simulation_results(
    filename_actual="actual.csv", filename_pred="predicted.csv"
)
