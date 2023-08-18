## Purpose
24 hour load forecasting using 15 minute load data.
## TimeSeriesLoader:
Provides a straightforward way to load the target variable and weather data from CSV files. It ensures that the file exists before loading and properly formats the index as a datetime type.

## TimeSeriesEDA:
Provides methods to plot the raw time series data and perform seasonal decomposition, allowing for visual inspection of the underlying patterns, trends, and seasonality.

## TimeSeriesPreprocessor:
Provides functionality for handling common preprocessing tasks in time series forecasting, including handling duplicates, aligning timestamps, filling missing indices, and handling missing and infinite values.

## TimeSeriesFeaturizer:
Provides methods for feature engineering specific to time series data. It includes functionality to create calendar features, such as month, day, and time of day, and to find optimal lags using PACF. These features are essential for capturing the temporal dependencies and patterns in time series forecasting.

## TimeSeriesXy:
Provides functionality to create future features for time series regression. It takes into account target variables and forecast horizons, allowing for flexibility in modeling different prediction tasks.

## TimeSeriesTrainer:
Provides a comprehensive solution for time series regression, including model training, hyperparameter tuning, cross-validation, and logging. It supports different cross-validation strategies and integrates with Optuna for hyperparameter optimization and MLflow for experiment tracking.

## TimeSeriesSimulator:
Provides functionality to simulate a production-like environment for time series forecasting. It includes methods to perform forecasting using a trained model and manage the test data and predictions.
