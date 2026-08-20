"""Walk-forward validation framework for evaluation"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PATH_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PATH_ROOT /"data" / "processed" / "model_dataset.csv"
OUTPUT_PATH = PATH_ROOT / "results" /"persistence"

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
        
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        # The persistance baseline does not require training 
        # Train set is constructed here so that the same walk-forward folds can latter be used for other models that require training
        train = df[df["date"] < test_start].copy()
        train = train.iloc[:-GAP_DAYS] # remove the last 5 days of the training set to avoid data leakage
        
        
        # print(test[[TARGET_COLUMN, BASELINE_COLUMN]].isnull().sum())
        # I tested, there's actually no missing values in this dataset, 
        # but just in case... 
        test = test.dropna(subset=[TARGET_COLUMN, BASELINE_COLUMN]) 
        
        y_true = test[TARGET_COLUMN]
        y_pred = test[BASELINE_COLUMN]
        
        MAE = mean_absolute_error(y_true, y_pred)
        RMSE = root_mean_squared_error(y_true, y_pred)
        
        # results : list of dictionaries
        # each dictionary contains the evaluation metrics for each test year 
        # Results will be saved to a daataframe and then to a csv file
        results.append({
            "test_year": y,
            "train_size": len(train),
            "test_size": len(test),
            "mae": MAE,
            "rmse": RMSE 
        })
        
        # for year y, fold_predictions (dataframe) contains for each date of this year :
        # the true value (volatility_5d), the predicted value (target_volatility_5d), 
        # the error and the absolute error
        fold_predictions = pd.DataFrame({
            "date": test["date"],
            # "test_year":y,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_true - y_pred,
            "abs_error": (y_true - y_pred).abs()
        })
        
        # predictions : list of dataframes
        # predictions will be saved to a single dataframe and then to a csv file
        predictions.append(fold_predictions)
        
    results = pd.DataFrame(results)
    predictions = pd.concat(predictions, ignore_index=True)
    
    return results, predictions

def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    results, predictions = walk_forward_evaluation(df)
    
    # Fold-level : results for each year
    print("Fold-level results:")
    print(results)
    
    print()
    
    # Mean of the fold-level results : average of the four years
    mean_mae = results["mae"].mean()
    mean_rmse = results["rmse"].mean()
    print(f"Mean fold MAE = {mean_mae:.6f}")
    print(f"Mean fold RMSE = {mean_rmse:.6f}")
    
    print()
    
    # Pooled-level : results for all dates combined, instead of averaging the fold-level results
    pooled_mae = mean_absolute_error(predictions["y_true"], predictions["y_pred"])
    pooled_rmse = root_mean_squared_error(predictions["y_true"], predictions["y_pred"])

    print(f"Pooled out-of-sample MAE: {pooled_mae:.6f}")
    print(f"Pooled out-of-sample RMSE: {pooled_rmse:.6f}")

    # Save the results and predictions from dataframes to csv files
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH / "fold_results.csv", index=False,)
    predictions.to_csv(OUTPUT_PATH / "predictions.csv", index=False,)


if __name__ == "__main__":
    main()        
        