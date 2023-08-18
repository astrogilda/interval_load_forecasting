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
# eda = TimeSeriesEDA()
# eda.perform_eda(y_data, target_variable="Load", freq=24, lags=40)  # type: ignore

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


"""
# Prepare data
X, y = TimeSeriesXy.df_to_X_y(df, "Load")

df_train, df_test = df.iloc[: -96 * 30 * 3], df.iloc[-96 * 30 * 3 :]

# Train model
trainer = TimeSeriesTrainer()
trained_model = trainer.train(df=df_train)

y_test, y_test_pred = TimeSeriesSimulator.forecast(
    df_test, "Load", trained_model
)

cv = SlidingWindowSplitter(
    window_length=96 * 30 * 3,
    start_with_window=True,
    step_length=96,
    fh=np.arange(1, 97),
)

for train_index, test_index in cv.split(df):
    print(f"TRAIN: {train_index}")
    print(f"TEST: {test_index}")
    print("")

# Preprocess data
X_train, y_train, X_test, y_test = data_loader.preprocess_data(
    y_data, weather_data
)

# Train model
model_trainer = TimeSeriesModel()
model = model_trainer.train_model(
    X_train, y_train, model_name="rr", metric_name="mae"
)

# Forecast
y_pred = model_trainer.forecast(model, X_test)

# Analyze and visualize
analyzer = TimeSeriesAnalysis()
residuals = y_test - y_pred
analyzer.plot_series(
    y_data, title="Time Series Data", xlabel="Time", ylabel="Value"
)
analyzer.plot_actual_vs_predicted(y_test, y_pred)
analyzer.plot_residuals(residuals)
analyzer.plot_residuals_heatmap(residuals)
"""
