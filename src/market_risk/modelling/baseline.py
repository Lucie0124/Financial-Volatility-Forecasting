"""Build the simplest benchmark baseline model"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from market_risk.features.build_target import TARGET_COLUMN, validate_target
from market_risk.features.volatility import annualized_realized_volatility
from market_risk.modelling.dataset import load_model_dataset

BASELINE_COLUMN = "persistence_volatility_5d"


def build_persistence_forecast(df: pd.DataFrame) -> pd.Series:
    """Predict future RMS volatility with the same measure over t-4 to t."""
    return annualized_realized_volatility(df["return_1d"]).rename(BASELINE_COLUMN)


def evaluate_baseline(df):
    validate_target(df)
    df = df.copy()

    df[BASELINE_COLUMN] = build_persistence_forecast(df)
    df = df.dropna(subset=[BASELINE_COLUMN, TARGET_COLUMN])

    y_true = df[TARGET_COLUMN]
    y_pred = df[BASELINE_COLUMN]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    print(f"Baseline mean absolute error MAE = {mae:4f}")
    print(f"Baseline root mean squared error RMSE = {rmse:4f}")


def main():
    df = load_model_dataset()
    evaluate_baseline(df)


if __name__ == "__main__":
    main()
