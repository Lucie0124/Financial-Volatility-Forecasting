"""Walk-forward validation framework for evaluation"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PATH_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PATH_ROOT /"data" / "processed" / "model_dataset.csv"

TEST_YEARS = [2022, 2023, 2024, 2025]

TARGET_COLUMN = "target_volatility_5d"
BASELINE_COLUMN = "volatility_5d"

GAP_DAYS = 5

def walk_forward_evaluation(df):
    df = df.copy()
    
    results = []
    
    for y in TEST_YEARS:
        
        # pd.Timestamp : the pandas equivalent of python’s Datetime
        test_start = pd.Timestamp(year=y, month=1, day=1)
        test_end = pd.Timestamp(year=y, month=12, day=31)
        
        train = df[df["date"] < test_start].copy()
        train = train.iloc[:-GAP_DAYS] # remove the last 5 days of the training set to avoid data leakage
        
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        
        print(test["date"].isna().sum())
        print(train["date"].isna().sum())
        

def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    
    results = walk_forward_evaluation(df)
    
    print(results)
    
if __name__ == "__main__":
    main()        
        