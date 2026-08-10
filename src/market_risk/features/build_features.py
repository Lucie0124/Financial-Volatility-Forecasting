"""Feature Engineering"""

from pathlib import Path
import pandas as pd
import numpy as np

PATH_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PATH_ROOT / "data" / "processed" / "market_data_clean.csv"
OUTPUT_PATH = PATH_ROOT / "data" / "processed" / "market_features.csv"

def build_features(df):
    df = df.copy()
    
    # 1. Equity returns
    df["return_1d"] = np.log(df["sp500"] / df["sp500"].shift(1))
    df["return_5d"] = np.log(df["sp500"] / df["sp500"].shift(5))
    
    # 2. Realised-volatility features
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()*np.sqrt(252)  # Annualized volatility
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()*np.sqrt(252)  
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()*np.sqrt(252)
    
    # 3. VIX features
    df["vix_change_1d"] = df["vix"].diff()
    df["vix_change_5d"] = df["vix"].diff(5)
    df["vix_vs_realized_vol"] = df["vix"]/100 - df["volatility_20d"] # VIX is in percentage points, so we divide by 100 to convert it to a decimal
    
    # 4. Intrest-rate features
    df["yield_curve"] = df["treasury_10y"] - df["treasury_2y"]
    df["treasury_10y_change_5d"] = df["treasury_10y"].diff(5) # 5-day change in 10-year treasury yield
    df["treasury_2y_change_5d"] = df["treasury_2y"].diff(5) # 5-day change in 2-year treasury yield
    
    return df


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = build_features(df)
    
    print(df.head())
    print()
    print("Missing values after feature engineering (%):")
    print((df.isnull().sum() / len(df) * 100).round(1))
    df.to_csv(OUTPUT_PATH, index=False)
    
    
if __name__ == "__main__":
    main()
