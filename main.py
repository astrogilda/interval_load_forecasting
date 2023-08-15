from timeseriesanalysis import TimeSeriesAnalysis
from timeserieseda import TimeSeriesEDA
from timeseriesloader import TimeSeriesLoader
from timeseriesmodel import TimeSeriesModel

# Load data
data_loader = TimeSeriesLoader(y_file="data/load.csv")
y_data, weather_data = data_loader.y_data, data_loader.weather_data

# Perform EDA
eda = TimeSeriesEDA()
eda.perform_eda(y_data, target_variable="Load", freq=24, lags=40)


"""
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
