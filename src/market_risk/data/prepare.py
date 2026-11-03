"""Market data alignment and cleaning pipeline"""

from pathlib import Path
import pandas as pd

PATH_ROOT = Path(__file__).resolve().parents[3]  
INPUT_PATH = PATH_ROOT / "data" / "processed" / "market_data.csv"
OUTPUT_PATH = PATH_ROOT / "data" / "processed" / "market_data_clean.csv"

INDICATOR_COLUMNS = ["vix", "treasury_10y", "treasury_2y", "high_yield_spread"]

def clean_data():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"]) # Parse the "date" column as datetime objects
    
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError("Market dates must be non-missing and unique.")

    # Drop rows where the S&P 500 index is missing and create a copy
    df = df.dropna(subset=["sp500"]).copy()  
    
    # Replace missing values with the most recently available (past) observation
    # for at most 5 consecutive rows
    df[INDICATOR_COLUMNS] = df[INDICATOR_COLUMNS].ffill(limit=5)  
    
    return df


def main():
    df = clean_data()
    
    print("Missing values after cleaning:")
    print(df.isna().sum())
    
    df.to_csv(OUTPUT_PATH, index=False)
    
if __name__ == "__main__":
    main()