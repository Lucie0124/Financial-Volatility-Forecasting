"""STress-regime evaluation"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from market_risk.modelling.dataset import TEST_YEARS, load_model_dataset, create_walk_forward_fold, prepare_train_test

ROOT_PATH = Path(__file__).resolve().parents[3]
RESULTS_PATH = ROOT_PATH/ "results"
OUTPUT_PATH = RESULTS_PATH/ "regime_evaluation"

NORMAL_QUANTILE = 0.7
STRESS_QUANTILE = 0.9


def get_regime_threshold(y_train):
    """define regime threshold"""
    elevated_threshold = y_train.quantile(NORMAL_QUANTILE)
    stress_threshold = y_train.quantile(STRESS_QUANTILE)
    return elevated_threshold, stress_threshold

# if y_train.quantile(0.7) returns 0.3 : 
# approximately 70% of training target were below 0.3


def classify_regime(volatility, elevated_threshold, stress_threshold):
    """Assign a regime"""
    
    if volatility >= stress_threshold:
        return "Stress"
    
    if volatility < stress_threshold and volatility >= elevated_threshold:
        return "Elevated"
    
    return "Normal"


def build_test_regimes(df):
    """build regime labels for each test year"""
    regime_rows = []
    
    for test_year in TEST_YEARS:
        train, test = create_walk_forward_fold(df, test_year)
        X_train, y_train, X_test, y_test = prepare_train_test(train, test)
        
        # define regime threshold with y_train
        elevated_threshold, stress_threshold = get_regime_threshold(y_train)
        
        
        # classify each date of y_test with a regime 
        fold_regimes = pd.DataFrame(
            {
                "date": test.loc[y_test.index, "date"],
                "y_true": y_test.values,
            }
        )
        fold_regimes["regime"] = (
            y_test.apply(
                lambda value: classify_regime(
                    value, 
                    elevated_threshold,
                    stress_threshold,
                )
            )
        )
        fold_regimes["elevated_threshold"] = elevated_threshold
        fold_regimes["stress_threshold"] = stress_threshold
        
        regime_rows.append(fold_regimes)
    
    # output DataFrame columns : ["date", "y_true", regime", "elevated_threshold", "stress_threshold"]
    regime_rows = pd.concat(regime_rows, ignore_index=True)
    
    return regime_rows


# Actually this function is not necessary, but it's just a way to simplify the next one
def load_predictions(model):
    """load each model's predictions"""
    path = RESULTS_PATH/ model/ "predictions.csv"
    
    predictions = pd.read_csv(path, parse_dates=["date"])
    predictions = predictions[["date", "y_pred", "error", "absolute_error"]].copy()
    
    # output DataFrame : ["date", "y_pred", "error", "absolute_error"]
    return predictions


def combine_predictions_with_regimes(regimes):
    """Combines model predictions with the regimes"""
     
    model_names = ["persistence", "ridge", "lightgbm"]
    combined = []
     
     
    for model in model_names:
        predictions = load_predictions(model)
        
        fold_merge = regimes.merge(predictions, on="date", how="inner")
        fold_merge["model"] = model
    
        combined.append(fold_merge)
    
    combined = pd.concat(combined, ignore_index=True)

    # output DataFrame : ["date", "y_pred", "error", "absolute_error", "y_true", regime", "elevated_threshold", "stress_threshold", "model"]
    return combined


def evaluate_by_regime(results):
    
    rows = []
    
    for (model, regime), group in results.groupby(["model", "regime"]):
        mae = mean_absolute_error(group["y_true"], group["y_pred"])
        rmse = root_mean_squared_error(group["y_true"], group["y_pred"])
        underestimation_rate = (group["error"] > 0).mean() # underestimated risk : when the actual volatility > forecast
        rows.append(
            {
                "model": model,
                "regime": regime,
                "count": len(group),
                "MAE": mae,
                "RMSE": rmse,
                "underestimation_rate": underestimation_rate,
            }
        )
    regime_metrics = pd.DataFrame(rows)
    
    regime_order = ["Normal", "Elevated", "Stress"]
    
    regime_metrics["regime"] = pd.Categorical(
        regime_metrics["regime"],
        categories=regime_order,
        ordered=True,
    )
        
    regime_metrics = regime_metrics.sort_values(["regime", "MAE"]).reset_index(drop=True)
    
    return regime_metrics


def main():
    df = load_model_dataset()
    regimes = build_test_regimes(df)
    pred_reg_combined = combine_predictions_with_regimes(regimes)
    
    regime_metrics = evaluate_by_regime(pred_reg_combined)

    
    
    print("Regime thresholds by year:")
    threshold_summary = (
        regimes[["date", "elevated_threshold", "stress_threshold"]]
        .assign(year=regimes["date"].dt.year)
        .groupby("year")
        [["elevated_threshold", "stress_threshold"]]
        .first()
    )
    
    print(threshold_summary.round(4))
    print()
    
    print("Model performance by regime")
    print(regime_metrics.round(4))
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    regimes.to_csv(OUTPUT_PATH/"regimes.csv")
    pred_reg_combined.to_csv(OUTPUT_PATH/"predictions_by_regime.csv", index=False)
    regime_metrics.to_csv(OUTPUT_PATH/"regime_metrics.csv", index=False)



if __name__ == "__main__":
    main()