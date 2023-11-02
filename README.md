[![DOI](https://zenodo.org/badge/681056609.svg)](https://zenodo.org/badge/latestdoi/681056609)

## Table of Contents

- [24 Hour Load Forecasting](#24-hour-load-forecasting)
  - [Overview](#overview)
  - [Modular Design](#modular-design)
  - [Forecasting Methodology](#forecasting-methodology)
  - [Future Enhancements](#future-enhancements)
- [Configuration Management](#configuration-management)
  - [`common_constants.py`](#common_constants.py)
  - [`time_constants.py`](#time_constants.py)
- [Data Loading](#data-loading)
  - [`time_series_loader.py`](#time_series_loader.py)
- [Data Preprocessing](#data-preprocessing)
  - [`time_series_preprocessor.py`](#time_series_preprocessor.py)
- [Feature Engineering](#feature-engineering)
  - [`time_series_featurizer.py`](#time_series_featurizer.py)
- [Data Structuring](#data-structuring)
  - [`time_series_xy.py`](#time_series_xy.py)
- [Model Training, Tracking, Hyperparameter Tuning, and Explainability](#model-training-tracking-hyperparameter-tuning-and-explainability)
  - [`time_series_trainer.py`](#time_series_trainer.py)
- [Time Series Simulation and Error Analysis](#time-series-simulation-and-error-analysis)
  - [`time_series_simulator.py`](#time_series_simulator.py)
- [Time Series Metrics Calculation](#time-series-metrics-calculation)
  - [`time_series_metrics.py`](#time_series_metrics.py)
- [Future Work](#future-work)

# 24 Hour Load Forecasting

This project focuses on using 15-minute load data for 24-hour load forecasting. It represents my efforts to align with software engineering best practices, following key principles where applicable. While it incorporates some essential practices, there is room for improvement and further refinement, providing an opportunity for collaboration and growth.

## Overview
24-hour load forecasting is a critical aspect of energy management, enabling accurate planning and optimization of energy resources. Utilizing 15-minute load data, this project employs commonly used machine learning techniques and methodologies to predict energy consumption patterns 24 hours ahead.

## Modular Design
The solution is structured into various modules, each responsible for specific tasks such as data loading, preprocessing, feature engineering, model training, simulation, and error analysis. This modular approach ensures flexibility, extensibility, and ease of maintenance.

## Forecasting Methodology
A variety of forecasting models are explored, including Random Forest, XGBoost, and Ridge Regression, along with hyperparameter tuning, cross-validation, and explainability techniques. The project also simulates a production environment to understand how the model performs with continuous data inflow.

## Future Enhancements
With a roadmap for future improvements, the project considers aspects such as continuous training, cloud deployment, API development, enhanced testing, and comprehensive documentation to ensure production readiness.

This project serves as a robust foundation for 24-hour load forecasting, aligning with industry standards and addressing the multifaceted challenges of time series forecasting in the energy domain.

--------------------------------------------------------------------------
## Configuration Management

The configuration management in this project is handled through two primary files: `common_constants.py` and `time_constants.py`. This approach is instrumental in avoiding magic numbers in the code, promoting clear understanding and maintainability. By centralizing these constants, the project facilitates rapid prototyping, allowing for efficient adjustments and scalability.

### `common_constants.py`
This file contains essential configurations and constants used throughout the project, including importing necessary libraries and dependencies, defining constants related to data loading, and importing time-related constants from the `time_constants.py` file.

### `time_constants.py`
This file focuses on defining time-related constants, essential for time series analysis and feature engineering. The constants include various time-related attributes like fifteen-minute intervals, seconds, minutes, hours, days, months, and years.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is an essential step in time series analysis that helps uncover underlying patterns, trends, and characteristics in the data. The methods provided in the `time_series_eda.py` file facilitate a comprehensive understanding of the time series data through various visualizations:

- **Time Series Plotting**: Visualize the raw time series data to identify potential seasonality, trends, and outliers.
- **Seasonal Decomposition**: Decompose the time series into seasonal, trend, and residual components to understand underlying patterns.
- **Autocorrelation Analysis**: Analyze the autocorrelations in the data, essential for model selection and tuning.
- **Distribution Analysis**: Understand the statistical distribution of the data and identify potential transformations.
- **Pattern Analysis**: Visualize specific patterns across different time frames (hourly, daily, monthly) to detect variations.
- **Heatmap Analysis**: Visualize relationships across grouped variables through heatmaps, providing insights into complex dependencies.

These analyses form the foundation for preprocessing, feature engineering, and model building, ensuring robust and accurate time series forecasting. The `time_series_eda.py` file contains methods for conducting exploratory data analysis (EDA) on time series data. Here's a detailed description of the main methods and their functionalities:

### 1. `plot_time_series` Method
Plots the time series data. Supports multiple time series with custom labels and colors. Plots can be saved to a file.

### 2. `plot_seasonal_decomposition` Method
Plots the seasonal decomposition using STL. Supports custom seasonalities and saving plots to a file.

### 3. `plot_acf_pacf` Method
Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). Supports saving plots to a file.

### 4. `plot_distribution` Method
Plots the distribution using a histogram and KDE. Supports saving plots to a file.

### 5. `plot_patterns` Method
Plots patterns using boxplots for different categories. Supports saving plots to a file.

### 6. `plot_heatmap` Method
Plots a heatmap by grouping the data by specified variables. Supports saving plots to a file.

Refer to the code documentation in `time_series_eda.py` for detailed information on parameters and usage.

## Data Loading

The `time_series_loader.py` file contains functionalities for loading time series data from CSV files:

### `TimeSeriesLoader` Class
- **Constructor**: Initializes with paths for y (target) data and weather data, loads the data.
- **`load_data` Method**: Loads data from CSV files and performs sanity checks, including file existence and data type validation.

Data loading plays a crucial role in gathering the raw time series data, ensuring its correctness and readiness for further analysis.

## Data Preprocessing

The `time_series_preprocessor.py` file contains functionalities for preprocessing time series data:

### `TimeSeriesPreprocessor` Class
- **Constructor**: Initializes with y (target) data and optional weather data.
- **`_handle_missing_indices` Method**: Fills missing indices, values, and infinite values in the data.
- **`_align_timestamps` Method**: Aligns timestamps across multiple data sources.
- **`merge_y_and_weather_data` Method**: Merges y data and weather data after handling inconsistencies and alignment.

Data preprocessing is essential to prepare the time series data for analysis, ensuring consistency, handling missing values, and alignment of different data sources.

## Feature Engineering

The `time_series_featurizer.py` file encompasses functionalities for transforming and enriching time series data through feature engineering. This process enhances the model's ability to capture complex temporal dynamics:

### `TimeSeriesFeaturizer` Class
- **`cyclical_encoding` Method**: Encodes cyclical features like months and days using sine and cosine transformations.
- **`create_calendar_features` Method**: Extracts calendar features such as month, day, hour, and minute.
- **`create_holiday_features` Method**: Adds binary features for US federal holidays.
- **`find_best_lag_pacf` Method**: Finds the optimal lag for a time series using the Partial Autocorrelation Function (PACF). It considers the maximum number of lags provided and returns the lag with the highest PACF, useful for identifying significant temporal relationships in the data.
- **`stats_features` Method**: Generates summary statistics features from a DataFrame. It computes mean, standard deviation, minimum, maximum, median, skewness, and kurtosis, providing a comprehensive statistical summary of the specified column, aiding in capturing the underlying distributional properties of the time series data.
- **`create_ar_features` Method**: Creates autoregressive features leveraging temporal dependencies.
- **`create_features` Method**: Prepares data for regression by combining various features and handling alignment, zero variability, and NaNs.

Feature engineering plays a pivotal role in revealing underlying patterns and relationships in the time series data, leading to robust and insightful models.

## Data Structuring

The `time_series_xy.py` file focuses on structuring time series data into a format suitable for modeling. This involves creating features that represent temporal dependencies, future horizons, and relevant attributes:

### `TimeSeriesXy` Class
- **`create_future_features` Method**: Generates future features for specified target variables at given forecast horizons. Includes validations for DateTimeIndex, target variables, and forecast horizons.
- **`df_to_X_y` Method**: Prepares the DataFrame for regression by creating features and structuring input (X) and target (y) variables.

Data structuring is vital in transforming raw time series data into a structured format, ready for modeling, ensuring that essential temporal relationships are captured and represented.

## Model Training, Tracking, Hyperparameter Tuning, and Explainability

The `time_series_trainer.py` file embodies a comprehensive approach to training, tuning, tracking, and explaining time series forecasting models:

### `TimeSeriesTrainer` Class
- **Constructor**: Initializes the trainer object.
- **`_calculate_and_log_shap_values` Method**: Calculates and logs SHAP values, saved as PNG plots and CSV files in `figures/results/shap` and `results/shap`.
- **`_create_cross_validator` Method**: Creates a cross-validator object based on specified strategies.
- **`train_model` Method**: Trains, tunes, and logs the model using cross-validation, hyperparameter optimization, feature scaling, and SHAP value calculation.

This section emphasizes robust model development, with a focus on optimization, tracking, and explainability. It facilitates an iterative and insightful modeling process, enabling a deeper understanding of time series dynamics and feature contributions.

### Files Generated
- **SHAP Summary Plots**: `figures/results/shap/shap_summary_step_{step}_model_{model_name}_{run_id}.png`
- **SHAP Values (CSV)**: `results/shap/shap_summary_step_{step}_model_{model_name}_{run_id}.csv`

### MLflow Integration

MLflow is integrated within the `TimeSeriesTrainer` class to facilitate experiment tracking, model logging, and the recording of hyperparameters and evaluation metrics:

- **Experiment Tracking**: Organizes and stores information related to different modeling runs, allowing for comparison and analysis of different models, hyperparameters, and results.
- **Model Logging**: Logs trained models, making them accessible for future use, sharing, and deployment.
- **Parameter and Metric Recording**: Captures model hyperparameters and evaluation metrics, enabling a transparent and traceable modeling process.

The use of MLflow enhances reproducibility, collaboration, and the overall effectiveness of the model development process. By systematically tracking and logging information, MLflow supports a data-driven approach to model selection, tuning, and explanation.

## Time Series Simulation and Error Analysis

The `time_series_simulator.py` file offers a comprehensive framework for simulating a production environment and analyzing errors in time series forecasting:

### `TimeSeriesRegression` Class
- **Constructor**: Initializes with combined y and weather data.
- **`forecast` Method**: Performs forecasting with the trained model.
- **`remove_duplicates` Method**: Removes duplicates from a DataFrame.
- **`simulate_production` Method**: Simulates a production environment, trains models, calculates residuals, and generates plots for analysis.

This simulation and error analysis approach provides insights into model performance in dynamic scenarios and a detailed understanding of error characteristics and distributions.

### Files Generated
- **Histogram of Mean Residuals**: `figures/residuals_histogram_mean.png`
- **Histogram of .95 Quantile Residuals**: `figures/residuals_histogram_p95quantile.png`
- **Histogram of .05 Quantile Residuals**: `figures/residuals_histogram_p05quantile.png`
- **Line Plot of Residuals Over Time**: `figures/residuals_lineplot.png`
- **Heatmap of Mean Residuals by Hour and Day**: `figures/residuals_heatmap.png`

## Time Series Metrics Calculation
The `time_series_metrics.py` file introduces a systematic approach to evaluate and compare different forecasting models. By calculating key metrics such as Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and Bias, this module provides quantitative insights into model performance. These metrics are instrumental in understanding the strengths and weaknesses of various models, allowing for informed decision-making and optimization.

### `TimeSeriesMetrics` Class
- **Constructor**: Initializes with the path of the CSV file containing the true values and a dictionary containing the names of models and paths of the CSV files containing the predicted values.
- **`_read_file` Method**: Reads data from a CSV file and returns it as a NumPy array.
- **`calculate_metrics` Method**: Calculates the MAE, MAPE, RMSE, and Bias for the given true and predicted values, returning a DataFrame containing the metrics.
- **`create_and_save_metrics` Method**: Creates and saves the calculated metrics to a specified CSV file.

This metrics calculation module facilitates a transparent and rigorous evaluation process, contributing to the project's goals of robustness, reproducibility, and collaboration. By encapsulating the metrics calculation within a dedicated class, the project maintains its modular design, ensuring flexibility and extensibility.

## Future Work

Several enhancements are required for production deployment:

- Logging and Monitoring: Replace print statements with proper logging to monitor model performance and errors.
- Scalability: Optimize code for parallel processing or distributed computing to handle large datasets.
- Configuration Management: Implement a configuration file to manage hyperparameters, paths, and other settings.
- Data Validation: Implement robust data validation to handle changes in data format or quality.
- Error Handling: Improve error handling to gracefully manage unexpected scenarios.
- Model Versioning: Integrate model versioning to manage different versions of trained models.
- Deployment Strategy: Plan for cloud deployment, considering aspects like security, scalability, and cost.
- Continuous Training: Implement continuous training to update the model as new data arrives, ensuring relevance.
- API Development: Develop APIs to allow other systems to consume the forecasts.
- Testing: Enhance test coverage, including unit and integration tests, to ensure code reliability.
- Documentation: Comprehensive documentation for maintainability and collaboration.
- Model Explainability and Retraining: Improve upong the current calculation of Shapley values, which are not directly applicaple to time-series data. Integrate this into a feature selection and model retraining pipeline.
