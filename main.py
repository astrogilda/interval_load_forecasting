from time_series_eda import TimeSeriesEDA
from time_series_loader import TimeSeriesLoader
from time_series_metrics import TimeSeriesMetrics
from time_series_preprocessor import TimeSeriesPreprocessor
from time_series_simulator import TimeSeriesSimulator

# Load data
data_loader = TimeSeriesLoader()
y_data, weather_data = data_loader.y_data, data_loader.weather_data

# Preprocess data
preprocessor = TimeSeriesPreprocessor(y_data, weather_data)
df = preprocessor.merge_y_and_weather_data(use_union=False)

# Simulate production
simulator = TimeSeriesSimulator(df)
simulator.simulate_production()

# Calculate and save metrics
y_true_file = "results/actual.csv"
y_pred_files = {
    "xgb": "results/predicted_model_xgb.csv",
    "rf": "results/predicted_model_rf.csv",
    "rr": "results/predicted_model_rr.csv",
}
output_file = "results/metrics.csv"
metrics = TimeSeriesMetrics(y_true_file, y_pred_files)
metrics.create_and_save_metrics(output_file)


# Perform EDA
# eda = TimeSeriesEDA
# eda.perform_eda(y_data, target_variable="Load")  # type: ignore
