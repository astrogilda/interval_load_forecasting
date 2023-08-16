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
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
)  # Number of lags to use for autoregressive features; 1 week
MAX_LAGS = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
)  # Maximum number of lags to use for autoregressive features; 1 week

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
OPTUNA_TRIALS = 100
HPO_FLAG = False  # Flag to enable hyperparameter optimization
CV_STRATEGY = "rolling"  # Cross-validation strategy. Must be one of: "rolling", "expanding"
MODEL_NAME = (
    "rr"  # Model name. Must be one of: "rr", "xgb", "lgbm", "rf", "mlp"
)
METRIC_NAME = "mae"  # Metric name. Must be one of: "mae", "mse", "rmse", "rmsle", "mape", "smape", "r2", "corr"

# Define metrics for Optuna objective function
OBJECTIVE_METRICS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "mape": mean_absolute_percentage_error,
}
# Define model hyperparameter spaces
MODEL_SPACES = {
    "rr": {"alpha": optuna.distributions.FloatDistribution(0.0, 1.0)},
    "rf": {
        "n_estimators": optuna.distributions.IntDistribution(2, 150),
        "max_depth": optuna.distributions.IntDistribution(1, 32),
    },
    "xgb": {
        "n_estimators": optuna.distributions.IntDistribution(2, 150),
        "max_depth": optuna.distributions.IntDistribution(1, 10),
        "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
    },
}
# Define model mapping
MODEL_MAPPING = {
    "rr": Ridge,
    "rf": RandomForestRegressor,
    "xgb": XGBRegressor,
}


# For simulating production
INITIAL_TRAIN_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH * 6
)  # 6 months
TEST_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
)  # 1 month
TRAIN_TEST_STEP_LENGTH = 24 * 7  # 1 week
