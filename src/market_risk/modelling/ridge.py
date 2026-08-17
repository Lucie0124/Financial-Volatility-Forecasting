"""Ridge regression with the built walk_forward folds"""

from pathlib import Path
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from market_risk.modelling.dataset import TEST_YEARS, create_walk_forward_fold, load_model_dataset, prepare_train_test

# Paths
PATH_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PATH_ROOT/ "results"/ "ridge"

# Initialization 
RIDGE_ALPHA = 1

# Build the model 

def build_ridge_model():
    model = Pipeline(steps=["scaler", StandardScaler()])

