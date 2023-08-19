import multiprocessing as mp

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

from time_constants import (
    DAYS_PER_MONTH,
    DAYS_PER_WEEK,
    FIFTEEN_MINUTES_PER_HOUR,
    HOURS_PER_DAY,
)

# For loading data
Y_FILE = "data/load.csv"  # Path of the CSV file containing y data
WEATHER_DATA_FILE = None  # Path of the CSV file containing weather data
TARGET_VARIABLE = "Load"  # Target variable name

# For creating features
AR_FROM_Y = True  # Autoregressive features from y
AR_FROM_WEATHER_DATA = False  # Autoregressive features from weather data
LAGS = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * 3
)  # Number of lags to use for autoregressive features; 3 days
MAX_LAGS = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * 3
)  # Maximum number of lags to use for autoregressive features; 3 days

# For creating X and y
FORECAST_HORIZON = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY
)  # Number of steps to forecast; 1 day


# For CV
WINDOW_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH * 3
)  # 3 months
INITIAL_WINDOW_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * HOURS_PER_DAY * 3
)  # 3 months
STEP_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
)  # 1 month
VAL_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
)  # 1 month
OPTUNA_TRIALS = N_TRIALS = 100
# Get the number of available CPU cores
num_cores = mp.cpu_count()
# Set the number of parallel jobs for HPO to 2/3 of the available cores
OPTUNA_JOBS = N_JOBS = int(num_cores * 2 / 3)
HPO_FLAG = False  # Flag to enable hyperparameter optimization
CV_STRATEGY = "rolling"  # Cross-validation strategy. Must be one of: "rolling", "expanding"
MODEL_NAME = "xgb"  # Model name. Must be one of: "rr", "xgb", "rf"
METRIC_NAME = "mae"  # Metric name. Must be one of: "mae", "mse", "rmse", "rmsle", "mape", "smape", "r2", "corr"

# Define metrics for Optuna objective function
OBJECTIVE_METRICS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "mape": mean_absolute_percentage_error,
}
# Define model hyperparameter spaces
MODEL_SPACES = {
    "rr": {
        "alpha": optuna.distributions.FloatDistribution(0.01, 1.0, log=True)
    },
    "rf": {
        "n_estimators": optuna.distributions.IntDistribution(2, 150),
        "max_depth": optuna.distributions.IntDistribution(1, 32),
        "min_samples_split": optuna.distributions.IntDistribution(2, 20),
        "min_samples_leaf": optuna.distributions.IntDistribution(1, 20),
    },
    "xgb": {
        "n_estimators": optuna.distributions.IntDistribution(2, 150),
        "max_depth": optuna.distributions.IntDistribution(1, 10),
        "learning_rate": optuna.distributions.FloatDistribution(
            0.01, 0.3, log=True
        ),
        "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
        "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
        "gamma": optuna.distributions.FloatDistribution(0, 5),
    },
}
# Define model mapping
MODEL_MAPPING = {
    "rr": Ridge,
    "rf": RandomForestRegressor,
    "xgb": XGBRegressor,
}
MLFLOW_LOGGING_FLAG = (
    True  # Flag to enable logging of parameters and metrics to MLflow
)
SHAP_VALUES_FLAG = False  # Flag to enable calculation of SHAP values


# For simulating production
INITIAL_TRAIN_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH * 6
)  # 6 months
TEST_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
)  # 1 month
TRAIN_TEST_STEP_LENGTH = 24 * 7  # 1 week
