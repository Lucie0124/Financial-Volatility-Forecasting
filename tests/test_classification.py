"""Testing the classification pipeline"""

# get_regime_threshold(...)
# classify_regime(...)
# walk-forward split boundaries
# 5-day purge between train and test
# classifier metric helper
# confirmation that test labels use training-fold thresholds only

# Note to self, use this in the terminal:
# pytest test_classification.py -v 

from market_risk.evaluation.regimes import combine_predictions_with_regimes

import numpy as np
import pytest
import pandas as pd

from market_risk.evaluation.regimes import (get_regime_threshold, classify_regime)
from market_risk.evaluation.classification import (compute_classification_metrics)


def test_regime_thresholds():
    y_train = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    elevated_threshold, stress_threshold = get_regime_threshold(y_train)
    assert elevated_threshold < stress_threshold


def test_classify_regime():
    elevated_threshold = 0.30
    stress_threshold = 0.50

    assert classify_regime(0.20, elevated_threshold, stress_threshold) == "Normal"
    assert classify_regime(0.40, elevated_threshold, stress_threshold) == "Elevated"
    assert classify_regime(0.60, elevated_threshold, stress_threshold) == "Stress"
    
    
# test for the metric helper
def test_classification_metrics():
    y_true = ["Normal", "Elevated", "Stress"]
    y_pred = ["Normal", "Elevated", "Stress"]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["stress_precision"] == 1.0
    assert metrics["stress_recall"] == 1.0
    assert metrics["stress_f1"] == 1.0

def test_stress_scores_do_not_report_the_normal_class():
    metrics = compute_classification_metrics(
        ["Normal"] * 4 + ["Elevated", "Stress", "Stress"],
        ["Normal"] * 4 + ["Stress", "Stress", "Elevated"],
    )
    assert metrics["stress_precision"] == pytest.approx(0.5)
    assert metrics["stress_recall"] == pytest.approx(0.5)
    assert metrics["stress_f1"] == pytest.approx(0.5)


def test_absent_stress_recall_is_undefined():
    metrics = compute_classification_metrics(["Normal"], ["Normal"])
    assert np.isnan(metrics["stress_recall"])


def test_threshold_equality_enters_the_higher_regime():
    assert classify_regime(0.3, 0.3, 0.5) == "Elevated"
    assert classify_regime(0.5, 0.3, 0.5) == "Stress"


@pytest.fixture
def regime_frames():
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    regimes = pd.DataFrame({
        "date": dates, "y_true": [0.1, 0.2], "regime": ["Normal", "Stress"]
    })
    predictions = pd.DataFrame({
        "date": dates, "y_true": [0.1, 0.2], "y_pred": [0.2, 0.1],
        "error": [999.0, 999.0], "absolute_error": [999.0, 999.0],
    })
    return regimes, predictions


def test_errors_are_recomputed_from_aligned_targets(regime_frames):
    regimes, predictions = regime_frames

    result = combine_predictions_with_regimes(regimes, {"ridge": predictions})

    assert len(result) == 2
    np.testing.assert_allclose(result["error"], [-0.1, 0.1])
    np.testing.assert_allclose(result["absolute_error"], [0.1, 0.1])


def test_legacy_prediction_targets_are_rejected(regime_frames):
    regimes, predictions = regime_frames
    predictions["y_true"] *= np.sqrt(5)

    with pytest.raises(ValueError, match="regenerate regression results"):
        combine_predictions_with_regimes(regimes, {"ridge": predictions})


@pytest.mark.parametrize("side", ["missing", "extra"])
def test_mismatched_dates_cannot_silently_disappear(regime_frames, side):
    regimes, predictions = regime_frames
    if side == "missing":
        predictions = predictions.iloc[:1]
    else:
        extra = predictions.iloc[[0]].assign(date=pd.Timestamp("2022-01-05"))
        predictions = pd.concat([predictions, extra], ignore_index=True)

    with pytest.raises(ValueError, match="dates do not match"):
        combine_predictions_with_regimes(regimes, {"ridge": predictions})


def test_duplicate_predictions_cannot_multiply_regime_rows(regime_frames):
    regimes, predictions = regime_frames
    predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)

    with pytest.raises(pd.errors.MergeError):
        combine_predictions_with_regimes(regimes, {"ridge": predictions})
