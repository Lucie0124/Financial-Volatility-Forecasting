"""Build the 5-day forward realised volatility target variable for the S&P 500 index"""

from pathlib import Path
import pandas as pd
import numpy as np

PATH_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PATH_ROOT / "data" / "processed" / "market_features.csv"
OUTPUT_PATH = PATH_ROOT / "data" / "processed" / "model_dataset.csv"

def build_target(df):
    df = df.copy()
    
    squared_returns = df["return_1d"] ** 2
    future_squared_returns = squared_returns.shift(-1).rolling(window=5).sum().shift(-4) # 
    df["target_volatility_5d"] = np.sqrt(future_squared_returns) * np.sqrt(252) 
    
    return df

def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = build_target(df)
    
    print(df[["date", "return_1d", "volatility_5d", "target_volatility_5d"]].tail(12))
    # the final 4 rows will have NaN values for the target variable since we cannot compute a 5-day forward volatility for them
    print()
    print("Missing target values:")
    print(df["target_volatility_5d"].isnull().sum())
    
    df.to_csv(OUTPUT_PATH, index=False)
    
if __name__ == "__main__":
    main()
    
    