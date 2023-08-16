from time_constants import (
    DAYS_PER_MONTH,
    DAYS_PER_WEEK,
    FIFTEEN_MINUTES_PER_HOUR,
    HOURS_PER_DAY,
)

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


# For simulating production
INITIAL_TRAIN_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH * 6
)  # 6 months
TEST_LENGTH = (
    FIFTEEN_MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
)  # 1 month
STEP_LENGTH = 24 * 7  # 1 week
