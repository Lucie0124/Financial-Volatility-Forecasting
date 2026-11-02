"""Shared non-centred realised-volatility definition for target and baseline."""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
VOLATILITY_WINDOW = 5


def annualized_realized_volatility(
    returns: pd.Series, window: int = VOLATILITY_WINDOW
) -> pd.Series:
    """Return sqrt(252 * mean(r**2)) over complete trailing windows.

    Values are annualised decimal volatilities. Returns are not demeaned;
    this is deliberately different from a rolling sample standard deviation.
    Input observations must be in chronological order. Missing returns make
    the corresponding window unavailable rather than shortening its horizon.
    """
    mean_squared_returns = returns.pow(2).rolling(
        window=window, min_periods=window
    ).mean()
    return np.sqrt(TRADING_DAYS_PER_YEAR * mean_squared_returns)
