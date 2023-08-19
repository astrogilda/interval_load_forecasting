# 24 Hour Load Forecasting

This project focuses on using 15-minute load data for 24-hour load forecasting. The solution is modular, maintainable, and follows best practices in software development and data science.

## Table of Contents

1. [Configuration Management] (#configuration-management)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Data Loading](#data-loading)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Data Structuring](#data-structuring)
7. [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
8. [Error Analysis](#error-analysis)
9. [Time Series Simulation](#time-series-simulation)
10. [Model Comparison](#model-comparison)
11. [Model Tracking and Explainability](#model-tracking-and-explainability)
12. [Future Work](#future-work)


## Configuration Management
`common_constants.py` and `time_constants.py`

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

Feature engineering is vital in capturing essential temporal patterns. The `TimeSeriesFeaturizer` class provides methods for feature engineering specific to time series data.

## Data Structuring

Proper data structuring aligns the data for modeling. The `TimeSeriesXy` class provides functionality to create future features for time series regression.

## Model Training, Tracking, Hyperparameter Tuning, and Explainability

Model training is the core of forecasting, involving model selection, tuning, and evaluation. The `TimeSeriesTrainer` class provides a comprehensive solution for time series regression, including model training, hyperparameter tuning, cross-validation, and logging. Tracking model versions and explaining their decisions is important in a machine learning pipeline. MLflow is integrated for model tracking, and SHAP (SHapley Additive exPlanations) is used for model explainability.

## Time Series Simulation and Error Analysis

Simulation emulates a production environment for validation. The `TimeSeriesSimulator` class provides functionality to simulate a production-like environment for time series forecasting.

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
