"""Walk-forward validation framework for evaluation"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from market_risk.modelling.baseline import build_persistence_forecast
from market_risk.modelling.dataset import (
    iter_walk_forward_folds,
    load_model_dataset,
)

PATH_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_PATH = PATH_ROOT / "results" /"persistence"


def walk_forward_evaluation(df):
    df = df.copy()
    # Compute on the full history before slicing years so the first test
    # prediction can use the final observed returns of the preceding year.
    forecast = build_persistence_forecast(df)
    
    results = []
    predictions = []
    
    for fold in iter_walk_forward_folds(df):
        y = fold.test_year
        train, test = fold.train, fold.test
        y_true = fold.y_test
        y_pred = forecast.loc[test.index]
        if y_pred.isna().any():
            raise ValueError(f"Fold {y} lacks a complete persistence window.")
        
        MAE = mean_absolute_error(y_true, y_pred)
        RMSE = root_mean_squared_error(y_true, y_pred)
        
        # results : list of dictionaries
        # each dictionary contains the evaluation metrics for each test year 
        # Results will be saved to a daataframe and then to a csv file
        results.append({
            "test_year": y,
            "train_size": len(train),
            "test_size": len(test),
            "MAE": MAE,
            "RMSE": RMSE 
        })
        
        # for year y, fold_predictions (dataframe) contains for each date of this year :
        # the target value, the trailing RMS-volatility prediction,
        # the error and the absolute error
        fold_predictions = pd.DataFrame({
            "date": test["date"],
            # "test_year":y,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_true - y_pred,
            "absolute_error": (y_true - y_pred).abs()
        })
        
        # predictions : list of dataframes
        # predictions will be saved to a single dataframe and then to a csv file
        predictions.append(fold_predictions)
        
    results = pd.DataFrame(results)
    predictions = pd.concat(predictions, ignore_index=True)
    
    return results, predictions

def main():
    df = load_model_dataset()
    
    results, predictions = walk_forward_evaluation(df)
    
    # Fold-level : results for each year
    print("Fold-level results:")
    print(results)
    
    print()
    
    # Mean of the fold-level results : average of the four years
    mean_mae = results["MAE"].mean()
    mean_rmse = results["RMSE"].mean()
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
