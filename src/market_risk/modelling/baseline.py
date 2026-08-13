"""Build the simplest benchmark baseline model"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PATH_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PATH_ROOT / "data" / "processed" / "model_dataset.csv"

def evaluate_baseline(df):
    df = df.copy()
    
    # keep only the rows where both the features and target are not missing
    df = df.dropna(subset=["volatility_5d", "target_volatility_5d"])
    
    y_true = df["target_volatility_5d"]
    
    # Persistent model: predict the next 5-day forward volatility as the current 5-day realized volatility
    y_pred = df["volatility_5d"]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    print(f"Baseline mean absolute error : {mae:4f}")
    print(f"Baseline root mean squared error: {rmse:4f}")
    

def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["dates"])
    evaluate_baseline(df)

if __name__ == "__main__":
    main()