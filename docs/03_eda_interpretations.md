# Exploratory data analysis

The [exploration notebook](../notebooks/01_data_exploration.ipynb) reads the
corrected target and verifies the saved experiment before displaying results.
All correlations are exploratory full-sample quantities; they are not used as
an independent out-of-sample feature-selection test.

The target is right-skewed and large observations cluster around a few stress
periods. Its maximum is **1.426833** (142.68% annualised) on **11 March 2020**.
The former claim of more than 300% used the incorrectly normalised target and
has been removed. Annualised units do not imply a one-year forecast horizon.

VIX has a strong positive association with future realised volatility, while
recent returns have a negative association. Neither establishes causality or
the incremental value of adding a predictor. The notebook calculates the
correlations and extreme-date tables directly from the current snapshot.

Historical-volatility features use rolling sample standard deviations with
`ddof=1`. They differ from the non-centred RMS used for target and persistence.
VIX is recorded in percentage points; `vix_vs_realized_vol` divides it by 100
before subtracting the 20-session volatility feature.

Consecutive five-session labels share four returns. Extreme dates therefore
cannot be counted as independent crises. The observed price trend alone is
not a formal stationarity test. Generalisation and forecasting value are
assessed separately in the chronological evaluation.
