# Market Volatility & Stress-Regime Forecasting

A learning project for forecasting S&P 500 realised volatility over the next
five trading sessions and classifying its future regime as Normal, Elevated
or Stress. The repository includes a reproducible historical experiment,
five executable notebooks and fitted model artifacts. An API and dashboard
remain future work.

## Definition and data

The annualised target uses five subsequent daily log returns:

$$
y_t=\sqrt{\frac{252}{5}\sum_{i=1}^{5}r_{t+i}^2},
\qquad r_t=\ln(P_t/P_{t-1}).
$$

Persistence applies the same non-centred RMS estimator to the five most
recent observed returns, including the current session. Rolling sample
standard deviations are separate predictors. Values are annualised decimal
volatilities; the observation horizon remains five sessions.

Daily S&P 500, VIX and 2Y/10Y Treasury series come from
[FRED](https://fred.stlouisfed.org/). The versioned snapshot covers
8 August 2016–31 July 2026. High-yield spread coverage is insufficient and
that column is excluded from the 14 model features. Raw downloads are ignored
by Git; processed CSVs are tracked. Re-downloading can change this snapshot.

## Install and reproduce

The reference environment is Python 3.12. `requirements-lock.txt` pins the
direct and transitive dependencies used for verification. From the repository
root, create and activate a virtual environment, then run:

```bash
python -m pip install -e ".[dev]" -c requirements-lock.txt
python -m market_risk.data.prepare
python -m market_risk.features.build_features
python -m market_risk.features.build_target
python -m market_risk.evaluation.run_evaluations
python -m market_risk.training.train_final_models
python -m market_risk.evaluation.provenance
python -m pytest -q
python scripts/execute_notebooks.py
```

These commands use the checked-in source snapshot without a network data
download. To deliberately refresh FRED inputs, run `python scripts/download_fred.py`
before the pipeline. Cleaning sorts dates before forward-filling at most five
observations and rejects duplicate dates.

Evaluation exports 16 CSVs and a completion manifest with data, source and
output fingerprints. Notebook analysis verifies these fingerprints. The
evaluation and final-training commands accept `--dataset` and `--output-dir`
for separate experiments. Notebook execution uses a fresh kernel per file;
`--output-dir path` saves executed copies elsewhere.

## Corrected results

An audit found a missing division by five in forward variance and an
inconsistent persistence estimator. The definitions, dataset, evaluations,
notebooks and artifacts have been corrected. Earlier outputs remain in Git
history and must not be combined with this experiment.


| model | count | MAE | RMSE |
| --- | --- | --- | --- |
| persistence | 1003 | 0.057393 | 0.086514 |
| ridge | 1003 | 0.054656 | 0.079227 |
| lightgbm | 1003 | 0.057531 | 0.083959 |

All models use the same 1,003 forecast dates in 2022–2025, with five origins
purged before each year. Ridge reduces pooled MAE by about **4.77%** against
the harmonised baseline. This replaces the misleading historical gain of
about 38%; no statistical significance is established.

In classification, Dummy has the highest pooled accuracy (0.6351), LightGBM
the highest macro-F1 (0.5062), and Logistic Regression the highest Stress
recall (0.8972). The latter detects 96 of 107 Stress dates with 201 false
Stress alerts. Recall is undefined in 2023 because there are no actual Stress
dates. The labels describe overlapping forecast dates, not independent crises.

## Interpretation and limits

Settings are fixed, not selected by a nested tuning procedure. The years
2022–2025 have already been inspected and do not form an untouched holdout.
Overlapping targets make adjacent losses dependent. Same-day indicator
availability at a precise execution time has not been verified. Probability
calibration, uncertainty estimates and economic decision costs remain open.
Final artifacts are fitted on all complete labels through 24 July 2026 and
are separate from the annual models used for evaluation.

See [methodology](docs/04_methodology.md), [results](docs/07_results.md),
[classification](docs/08_classification_analysis.md),
[reproduction](docs/09_reproducible_evaluation.md) and the
[correction record](docs/10_correction_record.md).

## AI-assisted documentation

AI assists with drafting documentation and reviewing code. Executable checks,
saved predictions and methodological explanations support independent review.
