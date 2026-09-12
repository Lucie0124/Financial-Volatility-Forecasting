"""Testing the classification pipeline"""

# get_regime_threshold(...)
# classify_regime(...)
# walk-forward split boundaries
# 5-day purge between train and test
# classifier metric helper
# confirmation that test labels use training-fold thresholds only

# pytest test_classification.py -v 

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