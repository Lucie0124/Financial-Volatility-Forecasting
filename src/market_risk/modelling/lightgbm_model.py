"""Implement a non-linear model LightGBM"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

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
    feature_importances = []
    
    for test_year in TEST_YEARS:
        
        train, test = create_walk_forward_fold(df, test_year)
        X_train, y_train, X_test, y_test = prepare_train_test(train, test)
        
        model = build_lightgbm_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # compute results
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        
        results.append(
            {
                "test_year": test_year,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "MAE": mae,
                "RMSE": rmse,
            }
        )
        
        # predictions
        fold_prediction = pd.DataFrame(
            {
                "date": test.loc[X_test.index, "date"],
                "y_true": y_test,
                "y_pred": y_pred,
                "error": y_test - y_pred,
                "absolute_error": (y_test - y_pred).abs()
            }
        )
        
        predictions.append(fold_prediction)

        # Feature importances
        fold_importance = pd.DataFrame(
            {
                "test_year": test_year,
                "feature": X_train.columns,
                "importance": model.feature_importances_ # how many times a feature was used to split tree nodes
            }
        )
        feature_importances.append(fold_importance)
        
    results = pd.DataFrame(results)
    predictions = pd.concat(predictions, ignore_index=True)
    feature_importances = pd.concat(feature_importances, ignore_index=True)
    
    return results, predictions, feature_importances


def main():
    df = load_model_dataset()
    
    results, predictions, feature_importances = evaluate_lightgbm_walk_forward(df)
    
    print("LightGBM fold-level results")
    print(results)
    
    print()
    mean_fold_mae = results["MAE"].mean()
    mean_fold_rmse = results["RMSE"].mean()
    print(f"Mean fold MAE: {mean_fold_mae:.6f}")
    print(f"Mean fold RMSE: {mean_fold_rmse:.6f}")
    
    print()
    pooled_mae = mean_absolute_error(predictions["y_true"], predictions["y_pred"])
    pooled_rmse = root_mean_squared_error(predictions["y_true"], predictions["y_pred"])
    print(f"Mean out-of-sample MAE: {pooled_mae:.6f}")
    print(f"Mean out-of-sample RMSE: {pooled_rmse:.6f}")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH/ "results.csv", index=False)
    predictions.to_csv(OUTPUT_PATH/ "predictions.csv", index=False)
    feature_importances.to_csv(OUTPUT_PATH/ "feature_importances.csv", index=False)

if __name__ == "__main__":
    main()