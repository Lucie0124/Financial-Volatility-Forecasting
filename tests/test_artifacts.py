import json
from pathlib import Path

import joblib
import hashlib
import numpy as np
import pytest

from market_risk.modelling.dataset import FEATURES, TARGET_COLUMN, load_model_dataset


ROOT_PATH = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = ROOT_PATH / "artifacts"


def test_artifacts_exist():
    assert (ARTIFACTS_PATH / "ridge_model.joblib").exists()
    assert (ARTIFACTS_PATH / "logistic_classifier.joblib").exists()
    assert (ARTIFACTS_PATH / "model_metadata.json").exists()


def test_versioned_artifacts_match_corrected_labels_and_training_population():
    metadata = json.loads((ARTIFACTS_PATH / 'model_metadata.json').read_text())
    df = load_model_dataset().dropna(subset=FEATURES + [TARGET_COLUMN])
    assert metadata['schema_version'] == 2
    assert metadata['training_rows'] == len(df)
    assert metadata['regime_thresholds']['stress'] == pytest.approx(df[TARGET_COLUMN].quantile(.9))
    for filename, checksum in metadata['artifact_sha256'].items():
        path = ARTIFACTS_PATH / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        model = joblib.load(path)
        np.testing.assert_allclose(model.named_steps['scaler'].mean_, df[FEATURES].mean())
        assert model.named_steps['scaler'].n_samples_seen_ == len(df)


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
