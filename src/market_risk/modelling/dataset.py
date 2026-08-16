"""Create a common modelling pipeline for all fitting models"""

from pathlib import Path
import pandas as pd

PATH_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PATH_ROOT / "data"/ "processed"/ "model_dataset.csv"

TARGET_COLUMN = "target_volatility_5d"

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

GAP_DAYS = 5


def load_model_dataset():
    """Loader function : ensures every model loads the data in exactly the same way"""
    df = pd.read_csv(INPUT_PATH, parse_dates = ["date"])
    df = df.sort_values("date").reset_index(drop=True) # discards old index
    return df

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


