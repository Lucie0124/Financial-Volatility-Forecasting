"""STress-regime evaluation"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from market_risk.modelling.dataset import TEST_YEARS, load_model_dataset, create_walk_forward_fold, prepare_train_test

ROOT_PATH = Path(__file__).resolve().parents[3]
RESULTS_PATH = ROOT_PATH/ "results"
OUTPUT_PATH = ROOT_PATH/ "regime_evaluation"

NORMAL_QUANTILE = 0.7
STRESS_QUANTILE = 0.9

def get_regime_threshold(y_train):
    """define regime threshold"""
    elevated_threshold = y_train.quantile(NORMAL_QUANTILE)
    stressed_threshold = y_train.quantile(STRESS_QUANTILE)
    return elevated_threshold, stressed_threshold

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
    """build regiem labels for each test year"""
    regime_rows = []
    
    for test_year in TEST_YEARS:
        train, test = create_walk_forward_fold(df)
        X_train, y_train, X_test, y_test = prepare_train_test(train, test)
        
        elevated_threshold, stressed_threshold = get_regime_threshold(y_train)
        
        fold_regimes = pd.DataFrame(
            {
                "date": test.loc[y_test.index, "date"],
                "y_true": y_test,
            }
        )
        fold_regimes["regime"] = (
            y_test.apply(
                lambda value: classify_regime(
                    value, 
                    elevated_threshold,
                    stressed_threshold,
                )
            )
        )
        fold_regimes["elevated_threshold"] = elevated_threshold
        fold_regimes["stressed_threshold"] = stressed_threshold
        
        regime_rows.append(fold_regimes)
    
    regime_rows = pd.concat(regime_rows, ignore_index=True)
    
    return regime_rows


def load_predictions(model):
    """load each model's predictions"""
    path = RESULTS_PATH/ model/ "predictions.csv"
    
    predictions = pd.read_csv(path, parse_dates="date")
    predictions = predictions[["date", "y_pred"]].copy()
    
    predictions["model"] = model
    
    return predictions


def combine_predictions_with_regimes(regimes):
     """Combines model predictions with the regimes"""
     
     model_names = ["peristance", "ridge", "lightgbm"]
     combined = []
     
     
     for model in model_names:
        predictions = load_predictions(model)
        
        model_results = regimes.merge(predictions, on="date", how="inner")
        model_results["model"]=  model
        
     
    