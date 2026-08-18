"""Implement a non-linear model LightGBM"""

from pathlib import Path
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from lightgbm import LGBMRegressor

from market_risk.modelling.dataset import TEST_YEARS, load_model_dataset, create_walk_forward_fold, prepare_train_test


ROOT_PATH = Path(__file__).resolve().parents[3]
OUTPUT_PATH = ROOT_PATH/ "results"/ "lightgbm"


# Model
def build_lightgbm_model():
    # tree models make decisions using thresholds (e.g. VIX<20)
    # so we don't need standardization unlike Ridge 
    # deliberately conservative starting parameters, not tuned
    model = LGBMRegressor(
        objective="regression",
        n_estimators=300,       # number of boosted trees
        learning_rate=0.03,     # boosting learning rate 
        max_depth=3,            # depth limit
        num_leaves=7,           # max number of leaves <= 2^max_depth
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    
    return model 

def evaluate_lightgbm_walk_forward(df):
    
    results = []
    predictions = []
    
    for test_year in TEST_YEARS:
        
        train, test = create_walk_forward_fold(df, test_year)
        X_train, y_train, X_test, y_test = prepare_train_test(train, test)
        
        model = build_lightgbm_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # compute results
        mae = mean_absolute_error(y_pred, y_test)
        rmse = root_mean_squared_error(y_pred, y_test)
        
        results.append(
            {
                "Test year": test_year,
                "Train size": len(X_train),
                "Test size": len(X_test),
                "MAE": mae,
                "RMSE": rmse,
            }
        )
        
        fold_prediction = pd.DataFrame(
            {
                "Date": test.loc[X_test.index, "date"],
                "y_true": y_test,
                "y_pred": y_pred,
                "Error": y_test - y_pred,
                "Absolute error": (y_test - y_pred).abs()
            }
        )
        
        predictions.append(fold_prediction)
    
    results = pd.DataFrame(results)
    predictions = pd.concat(predictions, ignore_index=True)
    
    return results, predictions 


def main():
    df = load_model_dataset()
    
    results, predictions = evaluate_lightgbm_walk_forward(df)
    
    print("LightGBM fold-level results")
    print(results)
    
    print()
    mean_fold_mae = results["mae"].mean()
    mean_fold_rmse = results["rmse"].mean()
    print(f"Mean fold MAE: {mean_fold_mae}:.6f")
    print(f"Mean fold RMSE: {mean_fold_rmse}:.6f")
    
    print()
    pooled_mae = mean_absolute_error(predictions["y_true"], predictions["y_pred"])
    pooled_rmse = root_mean_squared_error(predictions["y_true"], predictions["y_pred"])
    print(f"Mean out-of-sample MAE: {pooled_mae}:.6f")
    print(f"Mean out-of-sample RMSE: {pooled_rmse}:.6f")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH/ "lightgbm_results.csv", index=False)
    predictions.to_csv(OUTPUT_PATH/ "lightbgm_predictions.csv", index=False)


if __name__ == "__main__":
    main()