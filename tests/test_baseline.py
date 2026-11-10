"""Check that persistence predicts the same measure using only past returns"""

import numpy as np
import pandas as pd
import pytest

from market_risk.evaluation.walk_forward import walk_forward_evaluation
from market_risk.features.build_target import build_target
from market_risk.modelling.baseline import (
    build_persistence_forecast,
    evaluate_baseline,
)
from market_risk.modelling.dataset import FEATURES, load_model_dataset


def test_persistence_uses_current_and_four_previous_returns():
    df = pd.DataFrame(
        {"return_1d": [0.90, 0.01, -0.02, 0.03, -0.04, 0.05, -0.80]}
    )

    forecast = build_persistence_forecast(df)

    assert forecast.iloc[:4].isna().all()
    assert forecast.iloc[5] == pytest.approx(np.sqrt(252 * 0.0055 / 5))


def test_future_returns_do_not_change_a_persistence_prediction():
    df = pd.DataFrame({"return_1d": [0.01] * 12})
    changed = df.copy()
    changed.loc[8:, "return_1d"] = 0.90

    original = build_persistence_forecast(df)
    updated = build_persistence_forecast(changed)

    pd.testing.assert_series_equal(original.iloc[:8], updated.iloc[:8])
    assert updated.iloc[8] > original.iloc[8]


def test_persistence_requires_five_observed_returns():
    df = pd.DataFrame({"return_1d": [0.01] * 12})
    df.loc[5, "return_1d"] = np.nan

    forecast = build_persistence_forecast(df)

    assert forecast.iloc[5:10].isna().all()
    assert forecast.iloc[10:].notna().all()


def test_constant_returns_have_zero_persistence_error(capsys):
    df = build_target(pd.DataFrame({"return_1d": [0.01] * 12}))
    # The old sample-standard-deviation feature is zero for constant returns.
    # It remains a legitimate feature, but is not the RMS persistence forecast.
    df["volatility_5d"] = 0.0

    evaluate_baseline(df)

    output = capsys.readouterr().out
    assert "MAE = 0.000000" in output
    assert "RMSE = 0.000000" in output


def test_walk_forward_uses_the_same_rms_baseline_and_prior_year_history():
    dates = pd.bdate_range("2021-12-01", "2026-01-16")
    df = build_target(pd.DataFrame({"date": dates, "return_1d": 0.01}))
    for feature in FEATURES:
        if feature not in df:
            df[feature] = 0.0
    df["volatility_5d"] = 0.0
    original = df.copy(deep=True)

    metrics, predictions = walk_forward_evaluation(df)

    assert metrics["test_year"].tolist() == [2022, 2023, 2024, 2025]
    np.testing.assert_allclose(metrics[["MAE", "RMSE"]], 0.0, atol=1e-14)
    np.testing.assert_allclose(predictions["y_pred"], 0.15874507866387544)
    assert predictions["date"].iloc[0] == pd.Timestamp("2022-01-03")
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("evaluate", [evaluate_baseline, walk_forward_evaluation])
def test_baseline_evaluations_refuse_legacy_targets(evaluate):
    dates = pd.bdate_range("2021-12-01", periods=12)
    df = build_target(pd.DataFrame({"date": dates, "return_1d": 0.01}))
    df["target_volatility_5d"] *= np.sqrt(5)

    with pytest.raises(ValueError, match="Regenerate model_dataset.csv"):
        evaluate(df)


def test_model_loader_refuses_a_legacy_dataset(tmp_path, monkeypatch):
    dates = pd.bdate_range("2021-12-01", periods=12)
    df = build_target(pd.DataFrame({"date": dates, "return_1d": 0.01}))
    df["target_volatility_5d"] *= np.sqrt(5)
    path = tmp_path / "legacy_dataset.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr("market_risk.modelling.dataset.INPUT_PATH", path)

    with pytest.raises(ValueError, match="Regenerate model_dataset.csv"):
        load_model_dataset()


def test_model_loader_sorts_a_corrected_dataset(tmp_path, monkeypatch):
    dates = pd.bdate_range("2021-12-01", periods=12)
    expected = build_target(pd.DataFrame({"date": dates, "return_1d": 0.01}))
    for feature in FEATURES:
        if feature not in expected:
            expected[feature] = 0.0
    path = tmp_path / "corrected_dataset.csv"
    expected.iloc[::-1].to_csv(path, index=False)
    monkeypatch.setattr("market_risk.modelling.dataset.INPUT_PATH", path)

    result = load_model_dataset()

    assert result["date"].is_monotonic_increasing
    np.testing.assert_allclose(
        result["target_volatility_5d"],
        expected["target_volatility_5d"],
        equal_nan=True,
    )
