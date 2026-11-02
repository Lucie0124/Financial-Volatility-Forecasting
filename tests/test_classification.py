"""Testing the classification pipeline"""

# get_regime_threshold(...)
# classify_regime(...)
# walk-forward split boundaries
# 5-day purge between train and test
# classifier metric helper
# confirmation that test labels use training-fold thresholds only

# pytest test_classification.py -v 

import pandas as pd
import numpy as np
import pytest

from market_risk.evaluation.regimes import (get_regime_threshold, classify_regime)
from market_risk.evaluation.classification import (compute_classification_metrics)
from market_risk.evaluation.classification import compute_stress_probability_metrics


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
    assert classify_regime(elevated_threshold, elevated_threshold, stress_threshold) == 'Elevated'
    assert classify_regime(stress_threshold, elevated_threshold, stress_threshold) == 'Stress'
    
    
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


def test_stress_scores_are_not_the_normal_class_scores():
    actual = ["Normal"] * 4 + ["Elevated", "Stress", "Stress"]
    predicted = ["Normal"] * 4 + ["Stress", "Stress", "Elevated"]

    metrics = compute_classification_metrics(actual, predicted)

    assert metrics["accuracy"] == pytest.approx(5 / 7)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["stress_precision"] == pytest.approx(0.5)
    assert metrics["stress_recall"] == pytest.approx(0.5)
    assert metrics["stress_f1"] == pytest.approx(0.5)
    assert metrics["stress_support"] == 2
    assert metrics["predicted_stress_count"] == 2


def test_absent_stress_is_not_a_zero_detection_rate():
    metrics = compute_classification_metrics(["Normal", "Normal"], ["Normal", "Stress"])
    scores = compute_stress_probability_metrics(["Normal", "Normal"], [0.1, 0.8])

    assert metrics["stress_support"] == 0
    assert metrics["predicted_stress_count"] == 1
    assert np.isnan(metrics["stress_recall"])
    assert np.isnan(scores["stress_roc_auc"])
    assert np.isnan(scores["stress_average_precision"])
    # Both absent classes retain zero F1 in the fixed three-class average.
    assert metrics["macro_f1"] == pytest.approx((2 / 3) / 3)


def test_existing_stress_with_no_alerts_has_zero_recall():
    metrics = compute_classification_metrics(["Normal", "Stress"], ["Normal", "Normal"])

    assert metrics["stress_support"] == 1
    assert metrics["predicted_stress_count"] == 0
    assert metrics["stress_recall"] == 0.0


def test_dates_cannot_be_passed_as_an_extra_target():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_classification_metrics([["2022-01-03", "Normal"]], ["Normal"])


def test_perfect_stress_ranking_has_unit_auc_and_average_precision():
    result = compute_stress_probability_metrics(
        ["Normal", "Stress", "Elevated", "Stress"], [0.1, 0.9, 0.2, 0.8]
    )

    assert result["stress_roc_auc"] == 1.0
    assert result["stress_average_precision"] == 1.0


def test_all_positive_fold_has_no_roc_auc():
    result = compute_stress_probability_metrics(["Stress", "Stress"], [0.7, 0.9])

    assert np.isnan(result["stress_roc_auc"])
    assert result["stress_average_precision"] == 1.0
