"""Implement a non-linear model LightGBM"""

from pathlib import Path
import pandas as pd

from sklearn.metrics import mean_squared_error, root_mean_squared_error
from lightgbm import LGBMRegressor

from market_risk.modelling.dataset import TEST_YEARS, load_model_dataset, create_walk_forward_fold, prepare_train_test


ROOT_PATH = Path(__file__).resolve().parents[3]
OUTPUT_PATH = ROOT_PATH/ "results"/ "lightgbm"


# Model

def build_lightgbm_model():
    