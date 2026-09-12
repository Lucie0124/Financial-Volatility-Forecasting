import json
from pathlib import Path

import joblib


ROOT_PATH = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = ROOT_PATH / "artifacts"


def test_artifacts_exist():
    assert (ARTIFACTS_PATH / "ridge_model.joblib").exists()
    assert (ARTIFACTS_PATH / "logistic_classifier.joblib").exists()
    assert (ARTIFACTS_PATH / "model_metadata.json").exists()


def test_artifacts_can_be_loaded():
    ridge_model = joblib.load(
        ARTIFACTS_PATH / "ridge_model.joblib"
    )

    classifier = joblib.load(
        ARTIFACTS_PATH / "logistic_classifier.joblib"
    )

    with open(
        ARTIFACTS_PATH / "model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert ridge_model is not None
    assert classifier is not None
    assert "features" in metadata
    assert "regime_thresholds" in metadata


def test_artifacts_make_predictions():
    import json
    import joblib
    import pandas as pd

    ROOT_PATH = Path(__file__).resolve().parents[1]
    ARTIFACTS_PATH = ROOT_PATH / "artifacts"
    DATA_PATH = ROOT_PATH / "data" / "processed" / "model_dataset.csv"

    ridge_model = joblib.load(
        ARTIFACTS_PATH / "ridge_model.joblib"
    )

    classifier = joblib.load(
        ARTIFACTS_PATH / "logistic_classifier.joblib"
    )

    with open(
        ARTIFACTS_PATH / "model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    features = metadata["features"]

    valid_rows = df.dropna(
        subset=features
    )

    X = valid_rows[features].iloc[[0]]

    volatility_prediction = ridge_model.predict(X)

    regime_prediction = classifier.predict(X)

    regime_probabilities = classifier.predict_proba(X)

    assert len(volatility_prediction) == 1
    assert len(regime_prediction) == 1
    assert regime_prediction[0] in [
        "Normal",
        "Elevated",
        "Stress",
    ]

    assert regime_probabilities.shape == (1, 3)