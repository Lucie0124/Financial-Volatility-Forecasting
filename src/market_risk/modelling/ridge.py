"""Ridge regression with the built walk_forward folds"""

from pathlib import Path
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from market_risk.modelling.dataset import FEATURES, TEST_YEARS, create_walk_forward_fold, load_model_dataset, prepare_train_test

# Paths
PATH_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PATH_ROOT/ "results"/ "ridge"

# Initialization 
RIDGE_ALPHA = 1

# Build the model 
def build_ridge_model():
    model = Pipeline(steps=[  # Steps : an ordered list of transformations ending with an estimator/model
        ("scaler", StandardScaler()), # each tuple (name given to this step, object/transformation)
        ("ridge", Ridge(alpha=RIDGE_ALPHA)) # ("name", regression model)
        ])
    return model

# Walk-forward Ridge evaluation
def evaluate_ridge_walk_forward(df):
    
    results = []
    predictions = []
    coefficients = []
    
    for test_year in TEST_YEARS : 
        # initialisation
        train, test = create_walk_forward_fold(df, test_year)
        X_train, y_train, X_test, y_test = prepare_train_test(train, test)
        
        # model fitting
        model = build_ridge_model()
        model.fit(X_train, y_train)
        ridge_model = model.named_steps["ridge"]
        y_pred = model.predict(X_test)
        
        # Ridge coefifcents 
        fold_coefficients = pd.DataFrame(
            {
                "feature": FEATURES,
                "coefficient": ridge_model.coef_,
                "test_year": test_year
            }
        )
        
        coefficients.append(fold_coefficients)
        
        # results computation 
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        
        results.append({
            "test_year": test_year,
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "MAE" : mae,
            "RMSE" : rmse,
            })
        
        # predictions
        fold_predictions = pd.DataFrame(
            {
                "date": test.loc[X_test.index, "date"],
                "y_true": y_test,
                "y_pred": y_pred,
                "error": y_test - y_pred,
                "absolute_error": (y_test - y_pred).abs()
            }
        )
        
        predictions.append(fold_predictions)
    
    results = pd.DataFrame(results)
    predictions = pd.concat(predictions, ignore_index=True)
    coefficients = pd.concat(coefficients, ignore_index=False)
    
    return results, predictions, coefficients 
    
def main():
    df = load_model_dataset()
    results, predictions, coefficients = evaluate_ridge_walk_forward(df)
    
    print("Ridge fold_level results:")
    print(results)
    
    print()

    mean_fold_mae = results["MAE"].mean()
    mean_fold_rmse = results["RMSE"].mean()
    print(f"Mean fold MAE: {mean_fold_mae:.6f}")
    print(f"Mean fold RMSE: {mean_fold_rmse:.6f}")
    
    print()
    
    pooled_mae = mean_absolute_error(predictions["y_true"], predictions["y_pred"])
    pooled_rmse = root_mean_squared_error(predictions["y_true"], predictions["y_pred"])
    print(f"Pooled out-of-sample MAE: {pooled_mae:.6f}")
    print(f"Pooled out-of_sample RMSE: {pooled_rmse:.6f}")
    
    print()
    
    coefficient_summary = (
        coefficients
        .groupby("feature")["coefficient"]
        .agg(mean="mean", std="std", min="min", max="max")
    )
    coefficient_summary["abs_mean"] = coefficient_summary["mean"].abs()
    coefficient_summary = coefficient_summary.sort_values("abs_mean", ascending=False)
    print("Ridge coefficient summary:")
    print(coefficient_summary)
    
    # Save
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH/ "fold_results.csv", index=False)
    predictions.to_csv(OUTPUT_PATH/ "predictions.csv", index=False)
    coefficients.to_csv(OUTPUT_PATH/ "coefficients.csv", index=False)

if __name__ == "__main__":
    main()   
        
        
        
