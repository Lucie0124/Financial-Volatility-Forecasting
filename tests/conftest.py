"""Synthetic market histories; evaluation tests never rewrite repository data."""

import numpy as np
import pandas as pd
import pytest

from market_risk.features.build_features import build_features
from market_risk.features.build_target import build_target


@pytest.fixture
def corrected_dataset(monkeypatch):
    dates = pd.bdate_range("2021-08-02", "2023-02-28")
    steps = np.arange(len(dates))
    amplitude = np.array([0.002, 0.006, 0.012, 0.03])[(steps // 10) % 4]
    returns = amplitude * np.where(steps % 2, -1, 1)
    raw = pd.DataFrame({
        "date": dates,
        "sp500": 1000 * np.exp(np.cumsum(returns)),
        "vix": 12 + 1000 * amplitude,
        "treasury_10y": 2 + steps * 0.001,
        "treasury_2y": 1 + steps * 0.002,
    })
    monkeypatch.setattr("market_risk.modelling.dataset.TEST_YEARS", [2022, 2023])
    return build_target(build_features(raw))
