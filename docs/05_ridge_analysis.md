# Ridge regression

Ridge combines the 14 features after training-only standardisation and uses
L2 regularisation with alpha=1. Each year receives a newly fitted pipeline.
Its annual predictions, coefficients and metrics are exported under
`results/ridge/` by the [common runner](09_reproducible_evaluation.md).

| test_year | train_size | test_size | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 2022 | 1336 | 251 | 0.079577 | 0.103658 |
| 2023 | 1587 | 250 | 0.049507 | 0.063368 |
| 2024 | 1837 | 252 | 0.037805 | 0.050779 |
| 2025 | 2089 | 250 | 0.051770 | 0.088205 |

| model | count | MAE | RMSE |
| --- | --- | --- | --- |
| ridge | 1003 | 0.054656 | 0.079227 |

The pooled MAE gain against the corrected persistence benchmark is about
4.77%, substantially smaller than the historical 38% claim. The aggregate
gain is descriptive; no dependence-adjusted uncertainty or untouched holdout
establishes a general advantage.

Coefficients measure an effect per training standard deviation of a feature.
Correlated rolling volatilities and exact combinations such as yield_curve
and vix_vs_realized_vol make isolated coefficients hard to interpret. They
are not causal contributions or independent feature importance.

Stress MAE is 0.116293 and 75.70% of Stress targets are underestimated.
The [notebook](../notebooks/02_ridge_analysis.ipynb) displays the largest
errors, coefficient variation and actual-versus-predicted plots from current
exports. Ridge predictions are unconstrained and can be negative on other
inputs; this experiment does not silently clip forecasts or alter the model.
