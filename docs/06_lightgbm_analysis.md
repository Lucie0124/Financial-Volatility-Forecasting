# LightGBM regression

The fixed model uses 300 trees, learning_rate=0.03, max_depth=3,
num_leaves=7, min_child_samples=20 and reg_lambda=1. No scaling is required;
a new model is fitted before each test year with random_state=42.

| test_year | train_size | test_size | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 2022 | 1336 | 251 | 0.071660 | 0.090893 |
| 2023 | 1587 | 250 | 0.055495 | 0.065663 |
| 2024 | 1837 | 252 | 0.038997 | 0.052847 |
| 2025 | 2089 | 250 | 0.064065 | 0.113401 |

| model | count | MAE | RMSE |
| --- | --- | --- | --- |
| persistence | 1003 | 0.057393 | 0.086514 |
| ridge | 1003 | 0.054656 | 0.079227 |
| lightgbm | 1003 | 0.057531 | 0.083959 |

LightGBM has a slightly worse pooled MAE than persistence, while its pooled
RMSE is lower. Ridge has the lowest pooled MAE and RMSE. Complexity alone
does not establish value and the ordering varies with the metric and regime.

For realised Stress, LightGBM has MAE 0.112305, RMSE 0.162917 and an
underestimation rate of 61.68%. Its smaller underestimation rate can reflect
more overestimation and is not a measure of economic safety.

Feature importance exports count splits. Correlated variables can substitute
for one another, and counts neither measure causality nor prove incremental
predictive value. The [notebook](../notebooks/03_lightgbm_analysis.ipynb)
reads valid filenames, constructs comparisons before use and examines the
largest errors without claiming that a large forecast peak predicted a shock.
