"""Create a common modelling pipeline for all fitting models"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from market_risk.features.build_target import TARGET_COLUMN, validate_target
from market_risk.features.volatility import VOLATILITY_WINDOW

PATH_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PATH_ROOT / "data"/ "processed"/ "model_dataset.csv"

FEATURES = [
    "vix",
    "treasury_10y",
    "treasury_2y",
    
    "return_1d",
    "return_5d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "vix_change_1d",
    "vix_change_5d",
    "vix_vs_realized_vol",
    "yield_curve",
    "treasury_10y_change_5d",
    "treasury_2y_change_5d"
] 

TEST_YEARS = [2022, 2023, 2024, 2025]

GAP_DAYS = VOLATILITY_WINDOW


def load_model_dataset(path=None):
    """Loader function : ensures every model loads the data in exactly the same way"""
    df = pd.read_csv(INPUT_PATH if path is None else path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True) # discards old index
    validate_modelling_data(df)
    return df


def validate_modelling_data(df):
    """Check calendar, finite inputs and target alignment before any fitting."""
    dates = df['date']
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Modelling dates must be non-missing and unique.')
    if not dates.is_monotonic_increasing:
        raise ValueError('Modelling data must be sorted chronologically.')
    if not df.index.is_unique:
        raise ValueError('Modelling row indices must be unique.')
    validate_target(df)
    if np.isinf(df[FEATURES + [TARGET_COLUMN]].to_numpy(dtype=float)).any():
        raise ValueError('Modelling inputs must be finite or missing, never infinite.')

def create_walk_forward_fold(df, test_year):
    """Creates train, test for the test_year"""
    test_start = pd.Timestamp(year=test_year, month=1, day=1)
    test_end = pd.Timestamp(year=test_year, month=12, day=31)
    
    train = df[df["date"] < test_start].copy()
    train = train.iloc[:-GAP_DAYS] # their 5-day forward overlap with test period
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    
    return train, test 


def prepare_train_test(train, test):
    columns_needed = FEATURES + [TARGET_COLUMN]
    
    #print(train[required_columns].isna().sum())
    #print(test[required_columns].isna().sum())
    # No missing values in test but in train 
    
    train = train.dropna(subset=columns_needed).copy() # Remove rows containing missing values in these particular columns, but keep all original columns
    X_train = train[FEATURES]
    y_train = train[TARGET_COLUMN]
    
    test = test.dropna(subset=columns_needed).copy()
    X_test = test[FEATURES]
    y_test = test[TARGET_COLUMN]
    
    return X_train, y_train, X_test, y_test 


@dataclass(frozen=True)
class WalkForwardFold:
    """One purged fold with the same eligible rows for every model."""

    test_year: int
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def X_train(self):
        return self.train[FEATURES]

    @property
    def y_train(self):
        return self.train[TARGET_COLUMN]

    @property
    def X_test(self):
        return self.test[FEATURES]

    @property
    def y_test(self):
        return self.test[TARGET_COLUMN]


def iter_walk_forward_folds(df, test_years=None):
    """Validate the full dataset, then yield common complete-case folds.

    Validate labels before slicing: their forward returns can extend past a
    fold's final forecast date. Purge before dropping incomplete feature rows.
    """
    validate_modelling_data(df)

    years = list(TEST_YEARS if test_years is None else test_years)
    if not years or len(years) != len(set(years)):
        raise ValueError("Test years must be non-empty and unique.")
    for year in years:
        train, test = create_walk_forward_fold(df, year)
        X_train, _, X_test, _ = prepare_train_test(train, test)
        if X_train.empty or X_test.empty:
            raise ValueError(f"Fold {year} has no complete training or test rows.")
        yield WalkForwardFold(
            test_year=year,
            train=train.loc[X_train.index].copy(),
            test=test.loc[X_test.index].copy(),
        )
    

def main():
    df = load_model_dataset()
    for y in TEST_YEARS:
        train, test = create_walk_forward_fold(df, y)
        X_train, y_train, X_test, y_test = prepare_train_test(train, test)

        print(f"\nTest year: {y}")
        print("Training period: ", train["date"].min().date(), "-", train["date"].max().date())
        print("Test period: ", test["date"].min().date(), "-", test["date"].max().date())
        print("X_train shape: ", X_train.shape)
        print("X_test shape: ", X_test.shape)
        # y_train and y_test have the same shape as X, with just 1 column


if __name__ == "__main__":
    main()


