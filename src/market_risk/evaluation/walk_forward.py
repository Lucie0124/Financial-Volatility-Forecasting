"""Walk-forward validation framework for evaluation"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PATH_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PATH_ROOT /"data" / "processed" / "model_dataset.csv"
OUTPUT_PATH = PATH_ROOT / "data" / "processed"

TEST_YEARS = [2022, 2023, 2024, 2025]

TARGET_COLUMN = "target_volatility_5d"
BASELINE_COLUMN = "volatility_5d"

GAP_DAYS = 5

def walk_forward_evaluation(df):
    df = df.copy()
    
    results = []
    predictions = []
    
    for y in TEST_YEARS:
        
        # pd.Timestamp : the pandas equivalent of python’s Datetime
        test_start = pd.Timestamp(year=y, month=1, day=1)
        test_end = pd.Timestamp(year=y, month=12, day=31)
        
        train = df[df["date"] < test_start].copy()
        train = train.iloc[:-GAP_DAYS] # remove the last 5 days of the training set to avoid data leakage
        
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        
        # print(test[[TARGET_COLUMN, BASELINE_COLUMN]].isnull().sum())
        # I tested, there's actually no missing values in this dataset, 
        # but just in case... 
        test = test.dropna(subset=[TARGET_COLUMN, BASELINE_COLUMN]) 
        
        y_true = test[TARGET_COLUMN]
        y_pred = test[BASELINE_COLUMN]
        
        MAE = mean_absolute_error(y_true, y_pred)
        RMSE = root_mean_squared_error(y_true, y_pred)
        
        results.append({
            "test_year": y,
            "train_size": len(train),
            "test_size": len(test),
            "MAE": MAE,
            "RMSE": RMSE 
        })
        
        fold_predictions = pd.DataFrame({
            "date": test["date"],
            "test_year":y,
            "y_true": y_true,
            "y_pred": y_pred
        })
        
        fold_predictions["error"] = fold_predictions["y_true"] - fold_predictions["y_pred"]
        fold_predictions["abs_error"] = fold_predictions["error"].abs()
        
        predictions.append(fold_predictions)
        
      
    predictions = pd.concat(predictions, ignore_index=True)
    
    return pd.DataFrame(results), predictions

def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    results, predictions = walk_forward_evaluation(df)
    
    print("Fold-level results:")
    print(results)
    
    print()
    
    mean_mae = results["MAE"].mean()
    mean_rmse = results["RMSE"].mean()
    print(f"Mean walk_forward MAE = {mean_mae:.6f}")
    print(f"Mean walk_forward RMSE = {mean_rmse:6f}")
    
    print()
    
    pooled_mae = mean_absolute_error(predictions["y_true"], predictions["y_pred"])

    pooled_rmse = root_mean_squared_error(predictions["y_true"], predictions["y_pred"])

    print(f"Pooled out-of-sample MAE: {pooled_mae:.6f}")

    print(f"Pooled out-of-sample RMSE: {pooled_rmse:.6f}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_PATH / "persistence_fold_results.csv", index=False,)

    predictions.to_csv(OUTPUT_PATH / "persistence_predictions.csv", index=False,)

if __name__ == "__main__":
    main()        
        