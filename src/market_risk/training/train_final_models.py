"""Final model training and saving"""

# load the full modelling dataset
# prepare the final feature matrix
# train the selected Ridge regression model
# train the selected Logistic Regression regime classifier
# compute final regime thresholds from the available training targe
# save the fitted models and metadata for later use in the API/dashboar


import json 
from pathlib import Path
import joblib

from market_risk.evaluation.regimes import classify_regime, get_regime_threshold
from market_risk.modelling.dataset import FEATURES, TARGET_COLUMN, load_model_dataset
from market_risk.modelling.logistic_classifier import create_logistic_model
from market_risk.modelling.ridge import build_ridge_model

ROOT_PATH = Path(__file__).resolve().parents[3]
ARTIFACTS_PATH = ROOT_PATH / "artifacts"


def prepare_final_training_data(df):
    """Keep rows with complete features and target"""
    columns_needed = FEATURES + [TARGET_COLUMN]
    training_df = df.dropna(subset=columns_needed).copy()
    
    X = training_df[FEATURES]
    y = training_df[TARGET_COLUMN]
    
    return training_df, X, y


def train_final_ridge(X, y):
    """train the final Ridge regression model"""
    model = build_ridge_model()
    model.fit(X, y)
    
    return model


def train_final_classifier(X, y):
    """train the final volatility-regime classifier"""
    
    elevated_threshold, stress_threshold = get_regime_threshold(y)
    
    y_class = y.apply(lambda value: classify_regime(value, elevated_threshold, stress_threshold))
    model = create_logistic_model()
    model.fit(X, y_class)
    
    return model, elevated_threshold, stress_threshold


def save_artifacts(ridge_model, classifier_model, metadata):
    """Save fitted models and training metadata"""
    
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_model, ARTIFACTS_PATH / 'ridge_model.joblib')
    joblib.dump(classifier_model, ARTIFACTS_PATH / 'logistic_classifier.joblib')
    
    with open(ARTIFACTS_PATH / 'model_metadata.json', 'w', encoding='utf-8') as file: json.dump(metadata, file, indent=4)


def main():
    df = load_model_dataset()
    training_df, X, y = prepare_final_training_data(df)
    ridge_model = train_final_ridge(X, y)
    classifier_model, elevated_threshold, stress_threshold = train_final_classifier(X, y)
    
    metadata = {
        "features": FEATURES,
        "target": TARGET_COLUMN,
        "training_rows": len(training_df),
        "training_start": (training_df["date"].min().strftime("%Y-%m-%d")),
        "training_end": (training_df["date"].max().strftime("%Y-%m-%d")),
        "regime_thresholds": {
            "elevated": float(elevated_threshold),
            "stress": float(stress_threshold),
        },
        "regression_model": "Ridge",
        "classification_model": "Logistic Regression",
    }
    
    save_artifacts(ridge_model, classifier_model, metadata)
    
    print("Final models trained successfully.")
    print()
    print(f"Training observations: {len(training_df)}")
    print(
        "Training period:",
        training_df["date"].min().date(),
        "-",
        training_df["date"].max().date(),
    )
    print()
    print(
        f"Elevated threshold: "
        f"{elevated_threshold:.6f}"
    )
    print(
        f"Stress threshold: "
        f"{stress_threshold:.6f}"
    )
    print()
    print(f"Artifacts saved to: {ARTIFACTS_PATH}")
    
    
if __name__ == "__main__":
    main()
    