"""Scientific checks for the five-session forward volatility target."""

import numpy as np
import pandas as pd
import pytest

from market_risk.features.build_target import build_target, validate_target

def test_constant_one_percent_returns_have_known_annualized_volatility():
    df = pd.DataFrame({"return_1d": [0.01] * 12})

    result = build_target(df)

    # A non-centred RMS stays positive even when the sample std is zero.
    np.testing.assert_allclose(
        result["target_volatility_5d"].iloc[:-5],
        0.15874507866387544,
    )


def test_target_uses_exactly_the_next_five_returns():
    df = pd.DataFrame(
        {"return_1d": [np.nan, 0.01, -0.02, 0.03, -0.04, 0.05, 0.90]}
    )

    result = build_target(df)

    # Five future squared returns sum to 0.0055; the current and sixth
    # future returns are excluded, and no current return is required.
    assert result.loc[0, "target_volatility_5d"] == pytest.approx(
        np.sqrt(252 * 0.0055 / 5)
    )


def test_current_and_distant_future_returns_do_not_change_target():
    df = pd.DataFrame({"return_1d": [0.01] * 12})
    changed = df.copy()
    changed.loc[3, "return_1d"] = -0.80
    changed.loc[9, "return_1d"] = 0.90

    original_target = build_target(df).loc[3, "target_volatility_5d"]
    changed_target = build_target(changed).loc[3, "target_volatility_5d"]

    assert changed_target == pytest.approx(original_target)


def test_final_five_targets_are_missing():
    result = build_target(pd.DataFrame({"return_1d": [0.01] * 12}))

    assert result["target_volatility_5d"].iloc[:-5].notna().all()
    assert result["target_volatility_5d"].iloc[-5:].isna().all()


def test_zero_returns_give_zero_volatility():
    result = build_target(pd.DataFrame({"return_1d": [0.0] * 12}))

    assert (result["target_volatility_5d"].iloc[:-5] == 0.0).all()


def test_target_preserves_input_and_nondefault_index():
    df = pd.DataFrame(
        {"return_1d": [0.01] * 8},
        index=pd.date_range("2025-01-01", periods=8),
    )
    original = df.copy(deep=True)

    result = build_target(df)

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_index_equal(result.index, df.index)
    assert result["target_volatility_5d"].iloc[0] == pytest.approx(
        0.15874507866387544
    )


@pytest.mark.parametrize("length", [0, 1, 5])
def test_insufficient_future_history_has_no_target(length):
    df = pd.DataFrame({"return_1d": pd.Series([0.01] * length, dtype=float)})

    result = build_target(df)

    assert result["target_volatility_5d"].isna().all()


def test_missing_future_return_makes_the_full_horizon_unavailable():
    df = pd.DataFrame({"return_1d": [0.01] * 12})
    df.loc[5, "return_1d"] = np.nan

    result = build_target(df)

    assert result["target_volatility_5d"].iloc[:5].isna().all()
    assert result["target_volatility_5d"].iloc[5:7].notna().all()


def test_current_target_passes_definition_validation():
    df = build_target(pd.DataFrame({"return_1d": [0.01] * 12}))

    validate_target(df)


def test_legacy_sqrt_five_scaling_is_rejected():
    df = build_target(pd.DataFrame({"return_1d": [0.01] * 12}))
    df["target_volatility_5d"] *= np.sqrt(5)

    with pytest.raises(ValueError, match="Regenerate model_dataset.csv"):
        validate_target(df)


def test_filled_unavailable_targets_are_rejected():
    df = build_target(pd.DataFrame({"return_1d": [0.01] * 12}))
    df["target_volatility_5d"] = df["target_volatility_5d"].fillna(0.0)

    with pytest.raises(ValueError, match="Stored target values"):
        validate_target(df)
