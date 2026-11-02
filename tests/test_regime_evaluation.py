import numpy as np
import pandas as pd
import pytest

from market_risk.evaluation.regimes import combine_predictions_with_regimes


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
