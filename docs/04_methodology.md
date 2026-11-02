# Modelling methodology

At the end of session t, predictors use observations through t and the target
uses returns t+1 through t+5. This assumes daily indicators are available at
that decision time; exact publication and execution timestamps are not in the
snapshot, so operational availability remains unverified.

## Chronological folds

Training expands before each test year, 2022–2025. The last five training
forecast origins are removed before dropping incomplete feature rows, so
their forward labels cannot include test-period returns. No random split is
used. StandardScaler and regime thresholds are fitted on training rows only.
The shared generator uses all 14 features plus the target to select the same
complete cases for every regression and classification model.

| test_year | train_size | test_size |
| --- | --- | --- |
| 2022 | 1336 | 251 |
| 2023 | 1587 | 250 |
| 2024 | 1837 | 252 |
| 2025 | 2089 | 250 |

## Models

- Persistence: sqrt((252/5) times the sum of the last five squared log returns).
- Ridge: StandardScaler followed by Ridge(alpha=1).
- LightGBM regression: 300 trees, learning_rate=0.03, max_depth=3,
  num_leaves=7, min_child_samples=20, reg_lambda=1, random_state=42.
- Classification: majority Dummy, balanced Logistic Regression in a scaling
  pipeline, and balanced LightGBM with the same conservative tree settings.

The LightGBM models use four worker threads. Hyperparameters are fixed;
there is no nested tuning or new model selection in this correction.

## Metrics

Signed error is actual minus prediction. MAE is mean(abs(error)); RMSE is
sqrt(mean(error squared)). Both are in annualised decimal volatility units.
A MAE of 0.05 means five annualised volatility percentage points, not a 5%
investment loss. Pooled metrics are computed on all saved predictions, not
as averages of annual RMSE or F1 values.

Regimes use training-target quantiles q70 and q90. Values below q70 are
Normal, values from q70 up to q90 are Elevated, and values at or above q90
are Stress. Test regimes are realised labels; they are used ex post for error
analysis and as outcomes for the direct classifiers.

Classification exports fixed-three-class macro-F1, accuracy, Stress
precision/recall/F1, support, alert counts, ROC-AUC and average precision.
Undefined Stress recall is missing when support is zero. ROC-AUC requires
both outcomes; average precision is missing without positives. Undefined
precision and F1 use zero. See [classification](08_classification_analysis.md).

## Limits and final fitting

Purging the boundary prevents label overlap across train and test; it does
not remove dependence among adjacent test losses. No confidence interval,
significance claim, calibrated probability claim or economic backtest is
reported. The years have already been inspected for model comparison and
are not an untouched final holdout.

Final Ridge and Logistic Regression artifacts are refitted on all complete
labelled rows through 24 July 2026, with labels observed through 31 July.
Their final quantile thresholds differ from annual evaluation thresholds.
They are deployment candidates, not the models underlying the saved scores.
