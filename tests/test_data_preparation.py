import numpy as np
import pandas as pd
import pytest

from market_risk.data.prepare import INDICATOR_COLUMNS, clean_market_data
from market_risk.features.build_features import build_features
from market_risk.modelling.dataset import iter_walk_forward_folds
from market_risk.modelling.dataset import FEATURES


def test_cleaning_sorts_before_forward_fill_and_limits_gaps():
    df = pd.DataFrame({'date': pd.bdate_range('2022-01-03', periods=9), 'sp500': 100.0})
    for column in INDICATOR_COLUMNS:
        df[column] = [np.nan, 10.0] + [np.nan] * 6 + [99.0]
    shuffled = df.sample(frac=1, random_state=42)
    result = clean_market_data(shuffled)
    assert result['date'].is_monotonic_increasing
    assert pd.isna(result.loc[0, 'vix'])
    assert result.loc[2:6, 'vix'].eq(10.0).all()
    assert pd.isna(result.loc[7, 'vix'])
    pd.testing.assert_frame_equal(shuffled, df.sample(frac=1, random_state=42))


def test_cleaning_rejects_ambiguous_calendar():
    df = pd.DataFrame({'date': ['2022-01-03', '2022-01-03'], 'sp500': [100, 101]})
    with pytest.raises(ValueError, match='unique'):
        clean_market_data(df)


def test_feature_pipeline_rejects_reverse_calendar_and_invalid_prices(corrected_dataset):
    with pytest.raises(ValueError, match='chronologically'):
        build_features(corrected_dataset.iloc[::-1])
    changed = corrected_dataset.copy()
    changed.loc[0, 'sp500'] = 0
    with pytest.raises(ValueError, match='positive'):
        build_features(changed)


def test_all_models_reject_infinite_features(corrected_dataset):
    corrected_dataset.loc[25, 'vix'] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        list(iter_walk_forward_folds(corrected_dataset))


def test_future_observations_do_not_change_present_or_past_features(corrected_dataset):
    original = build_features(corrected_dataset)
    changed = corrected_dataset.copy()
    cutoff = changed['date'].iloc[80]
    future = changed['date'] > cutoff
    changed.loc[future, 'sp500'] *= 2
    changed.loc[future, 'vix'] += 30
    changed.loc[future, 'treasury_10y'] += 1
    changed.loc[future, 'treasury_2y'] -= 1
    rebuilt = build_features(changed)
    pd.testing.assert_frame_equal(original.loc[~future, FEATURES], rebuilt.loc[~future, FEATURES])
