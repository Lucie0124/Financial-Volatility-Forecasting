# Corrected historical results

The common evaluation uses 1,003 forecast dates in 2022–2025. All models share
eligible rows and a five-origin training purge. Target and persistence now
use symmetric annualised five-session RMS definitions. The dataset, results
and artifacts replace the inconsistent experiment; old files remain in Git
history. See [reproduction](09_reproducible_evaluation.md).

## Pooled regression

| model | count | MAE | RMSE |
| --- | --- | --- | --- |
| persistence | 1003 | 0.057393 | 0.086514 |
| ridge | 1003 | 0.054656 | 0.079227 |
| lightgbm | 1003 | 0.057531 | 0.083959 |

Ridge reduces MAE by about 4.77% and RMSE by about 8.42% relative to persistence.
LightGBM improves RMSE but does not improve MAE against this baseline.
The previous gain of about 38% was affected by inconsistent measurement.
Metrics are annualised decimal volatility errors, not investment returns.

## Annual results

| model | test_year | train_size | test_size | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| persistence | 2022 | 1336 | 251 | 0.076514 | 0.099428 |
| persistence | 2023 | 1587 | 250 | 0.039952 | 0.050550 |
| persistence | 2024 | 1837 | 252 | 0.046372 | 0.059919 |
| persistence | 2025 | 2089 | 250 | 0.066746 | 0.118019 |
| ridge | 2022 | 1336 | 251 | 0.079577 | 0.103658 |
| ridge | 2023 | 1587 | 250 | 0.049507 | 0.063368 |
| ridge | 2024 | 1837 | 252 | 0.037805 | 0.050779 |
| ridge | 2025 | 2089 | 250 | 0.051770 | 0.088205 |
| lightgbm | 2022 | 1336 | 251 | 0.071660 | 0.090893 |
| lightgbm | 2023 | 1587 | 250 | 0.055495 | 0.065663 |
| lightgbm | 2024 | 1837 | 252 | 0.038997 | 0.052847 |
| lightgbm | 2025 | 2089 | 250 | 0.064065 | 0.113401 |

## Errors by future realised regime

| model | regime | count | MAE | RMSE | underestimation_rate |
| --- | --- | --- | --- | --- | --- |
| ridge | Normal | 637 | 0.041742 | 0.056252 | 0.238619 |
| persistence | Normal | 637 | 0.043056 | 0.060563 | 0.406593 |
| lightgbm | Normal | 637 | 0.047964 | 0.061248 | 0.185243 |
| lightgbm | Elevated | 259 | 0.058434 | 0.084302 | 0.459459 |
| ridge | Elevated | 259 | 0.060954 | 0.076990 | 0.613900 |
| persistence | Elevated | 259 | 0.065414 | 0.090634 | 0.602317 |
| lightgbm | Stress | 107 | 0.112305 | 0.162917 | 0.616822 |
| ridge | Stress | 107 | 0.116293 | 0.160165 | 0.757009 |
| persistence | Stress | 107 | 0.123334 | 0.168641 | 0.766355 |

Regimes are based on the realised future target and are diagnostic labels,
not information available at forecast time. A smaller underestimation rate
does not establish better decisions without measuring overestimation costs.

## Classification and limits

Dummy attains pooled accuracy 0.635095, above both fitted classifiers.
LightGBM attains the highest macro-F1, 0.506225. Logistic Regression detects
96/107 Stress dates and emits 201 false Stress alerts. Both fitted classifiers
miss the five Stress dates in 2024; no actual Stress occurs in 2023.
See [complete classification metrics](08_classification_analysis.md).

Stress is concentrated in 2022 (85 of 107 dates). Overlapping labels create
dependent losses and repeated dates within episodes. No confidence intervals,
probability calibration, untouched final holdout or economic backtest supports
a production-performance claim. The final artifacts use data through 2026
and must not be used to recreate earlier years as out-of-sample predictions.
