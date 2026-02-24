"""
Centralized Configuration Module

This module contains all configuration parameters, paths, and constants
used throughout the project. Centralizing configuration ensures consistency,
maintainability, and easy modification of project settings.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "Flight_Price_Dataset_of_Bangladesh.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model files
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
FEATURE_ENGINEER_FILE = MODELS_DIR / "feature_engineer.joblib"
MODEL_METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Logging configuration
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data preprocessing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MISSING_VALUE_STRATEGY = {
    'numerical': 'median',
    'categorical': 'mode'
}

# Feature engineering parameters
DATE_FEATURES = ['month', 'day', 'weekday', 'season']
CATEGORICAL_ENCODING = 'onehot'  # Options: 'onehot', 'label'
NUMERICAL_SCALING = 'standard'    # Options: 'standard', 'minmax', 'robust'

# Feature categorization for model-optimized encoding
# Numerical (Continuous): Duration, Days Before - scaled for Linear/Ridge/Lasso
# Ordinal (Ranked): Class - Economy=0, Business=1, First Class=2
# Categorical (Nominal): Airline, Source, Destination, Seasonality - OneHot encoded
# Binary (Flag): Stopovers - Direct=0, Any Stop=1
# Derived (Temporal): Dep_Hour, Dep_Weekday, Dep_TimeBin - extracted from datetime
ORDINAL_CLASS_MAP = {'Economy': 0, 'Business': 1, 'First Class': 2}
HOUR_BINS = [-1, 5, 11, 17, 23]  # Early Morning, Morning, Afternoon, Evening
HOUR_BIN_LABELS = [0, 1, 2, 3]

# Model training parameters
CV_FOLDS = 5
SCORING_METRIC = 'neg_mean_squared_error'

# Model hyperparameters for GridSearch
RIDGE_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

LASSO_PARAMS = {
    'alpha': [0.01, 0.1, 1.0, 10.0]
}

DECISION_TREE_PARAMS = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

XGBOOST_PARAMS = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

LIGHTGBM_PARAMS = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63],
    'feature_fraction': [0.8, 1.0],
    'bagging_fraction': [0.8, 1.0]
}

# Visualization parameters
FIGURE_SIZE = (12, 6)
DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'