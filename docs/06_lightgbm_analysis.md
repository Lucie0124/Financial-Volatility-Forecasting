# LightGBM Analysis and Model Comparison

## Overview

After evaluating the persistence baseline and Ridge regression, the project introduces **LightGBM** as a nonlinear forecasting model.

The main objective is to answer:

> Can a nonlinear tree-based model improve five-day forward realised-volatility forecasts beyond the linear Ridge model?

The progression of the modelling experiment is:

```text
Persistence
    ↓
Uses recent 5-day realised volatility only

Ridge
    ↓
Combines all features through linear relationships

LightGBM
    ↓
Captures nonlinear relationships,
threshold effects, and feature interactions
```

All three models are evaluated using the same chronological walk-forward framework.

---

# Why LightGBM?

LightGBM is a gradient-boosted decision-tree model.

Unlike Ridge regression, it does not assume that the relationship between each feature and the target is linear.

For example, Ridge effectively assumes that the effect of VIX can be approximated by:

```text
higher VIX
    ↓
proportional change in predicted volatility
```

LightGBM can instead learn threshold-type behaviour such as:

```text
VIX < 15
→ low-volatility regime

15 <= VIX < 30
→ elevated regime

VIX >= 30
→ stress regime
```

It can also capture interactions.

For example:

```text
VIX high
AND
recent return strongly negative
AND
recent realised volatility elevated
```

may produce a different prediction from any of these conditions individually.

This flexibility makes LightGBM a useful nonlinear benchmark.

---

# LightGBM Configuration

The initial model is intentionally conservative and untuned.

The configuration is approximately:

```python
LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    num_leaves=7,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)
```

The objective is not to search aggressively for the best hyperparameters.

Instead, the first experiment asks:

> Does a reasonable nonlinear model already outperform Ridge?

Hyperparameter tuning can later be performed using training data only.

---

# Why No Feature Standardisation?

Unlike Ridge regression, LightGBM does not require `StandardScaler`.

Tree-based models split on thresholds.

For example:

```text
VIX < 22.5 ?
```

If VIX were standardised, the numerical threshold would change, but the ordering of observations would remain the same.

The model would still be able to construct equivalent splits.

Therefore:

```text
Ridge
→ scaling required

LightGBM
→ scaling not required
```

---

# Walk-Forward Evaluation

LightGBM uses the same expanding-window evaluation as the previous models.

The folds are:

```text
Fold 1
Train: historical data before 2022
Test:  2022

Fold 2
Train: historical data before 2023
Test:  2023

Fold 3
Train: historical data before 2024
Test:  2024

Fold 4
Train: historical data before 2025
Test:  2025
```

The final five training observations are purged before each test period to prevent overlap between training targets and test-period returns.

For each fold:

1. the model is trained from scratch;
2. the test year remains completely out of sample;
3. MAE and RMSE are calculated;
4. predictions are saved for later diagnostic analysis.

---

# LightGBM Results

## Fold-Level Performance

The LightGBM walk-forward results are:

| Test Year | Train Size | Test Size | MAE | RMSE |
|---|---:|---:|---:|---:|
| 2022 | 1336 | 251 | 0.1602 | 0.2032 |
| 2023 | 1587 | 250 | 0.1241 | 0.1468 |
| 2024 | 1837 | 252 | 0.0872 | 0.1182 |
| 2025 | 2089 | 250 | 0.1433 | 0.2536 |

The mean fold performance is:

```text
Mean fold MAE  = 0.1287
Mean fold RMSE = 0.1805
```

The pooled out-of-sample performance is:

```text
Pooled MAE  = 0.1286
Pooled RMSE = 0.1877
```

---

# Comparison with Persistence and Ridge

The pooled out-of-sample results are:

| Model | Pooled MAE | Pooled RMSE |
|---|---:|---:|
| Persistence | 0.1982 | 0.2574 |
| Ridge | **0.1222** | **0.1772** |
| LightGBM | 0.1286 | 0.1877 |

The overall ranking is therefore:

```text
1. Ridge
2. LightGBM
3. Persistence
```

LightGBM substantially improves on the persistence baseline.

However, Ridge remains the best-performing model on both pooled MAE and pooled RMSE.

---

# Approximate Improvement Relative to Persistence

The persistence baseline achieves:

```text
MAE  = 0.1982
RMSE = 0.2574
```

LightGBM achieves:

```text
MAE  = 0.1286
RMSE = 0.1877
```

This corresponds approximately to:

```text
35% reduction in MAE
27% reduction in RMSE
```

relative to persistence.

This confirms that combining multiple market indicators provides substantial predictive value beyond recent five-day realised volatility alone.

---

# Ridge vs LightGBM

The Ridge model achieves:

```text
Pooled MAE  = 0.1222
Pooled RMSE = 0.1772
```

while LightGBM achieves:

```text
Pooled MAE  = 0.1286
Pooled RMSE = 0.1877
```

Ridge therefore performs better overall.

This is an important result because the more complex nonlinear model does not automatically outperform the simpler linear model.

The result suggests that much of the predictive information contained in the current feature set may already be represented effectively through relatively stable linear relationships.

---

# Year-by-Year Model Comparison

The MAE comparison is:

| Test Year | Persistence | Ridge | LightGBM |
|---|---:|---:|---:|
| 2022 | 0.2943 | 0.1779 | **0.1602** |
| 2023 | 0.1565 | **0.1107** | 0.1241 |
| 2024 | 0.1509 | **0.0845** | 0.0872 |
| 2025 | 0.1911 | **0.1158** | 0.1433 |

This shows that the relative ranking changes across market periods.

---

## 2022

LightGBM performs best in 2022:

```text
Persistence MAE = 0.2943
Ridge MAE       = 0.1779
LightGBM MAE    = 0.1602
```

This suggests that nonlinear relationships or feature interactions were particularly useful during this period.

LightGBM appears to capture some volatility dynamics that Ridge cannot represent as effectively with a linear structure.

---

## 2023

Ridge performs best in 2023:

```text
Ridge MAE      = 0.1107
LightGBM MAE   = 0.1241
```

The simpler linear model generalises better during this test period.

This suggests that the additional flexibility of LightGBM does not always translate into better out-of-sample accuracy.

---

## 2024

The two models are very close:

```text
Ridge MAE      = 0.0845
LightGBM MAE   = 0.0872
```

Both perform substantially better than persistence.

This indicates that both linear and nonlinear models capture the dominant volatility signals well during this period.

---

## 2025

Ridge performs substantially better:

```text
Ridge MAE      = 0.1158
LightGBM MAE   = 0.1433
```

The RMSE difference is even larger:

```text
Ridge RMSE      = 0.1972
LightGBM RMSE   = 0.2536
```

The higher LightGBM RMSE indicates that a small number of very large errors occur during this year.

These errors are examined in detail below.

---

# LightGBM Feature Importance

LightGBM feature importance was collected for each walk-forward fold.

The current analysis uses split importance.

This measures how frequently each feature is used to split tree nodes.

It does not measure the sign of the relationship.

For example:

```text
high importance
```

means:

> the feature is used frequently by the tree ensemble.

It does not mean:

```text
higher feature value
→ higher prediction
```

and it does not imply a causal relationship.

---

# Main Important Features

The features used most frequently by LightGBM include:

```text
VIX
volatility_20d
volatility_10d
yield_curve
volatility_5d
return_5d
```

The dominant variables again belong primarily to:

```text
option-implied volatility
+
recent realised volatility
```

This is consistent with both the exploratory analysis and Ridge regression.

---

# VIX Importance

VIX is one of the most important LightGBM features across the walk-forward folds.

This provides consistency across three stages of the project:

```text
EDA
→ VIX has the strongest correlation with the target

Ridge
→ VIX receives one of the largest positive coefficients

LightGBM
→ VIX is one of the most frequently used splitting variables
```

This strengthens the conclusion that VIX contains substantial information about future realised volatility.

---

# Realised-Volatility Features

`volatility_20d` and `volatility_10d` also receive large feature importance.

This is consistent with volatility persistence.

Recent volatility conditions contain useful information about subsequent volatility.

LightGBM may use these variables through nonlinear thresholds such as:

```text
volatility_20d < threshold?
```

rather than assigning a single linear coefficient.

---

# Yield Curve Importance

One interesting difference from Ridge is the relatively high LightGBM split importance of:

```text
yield_curve
```

The yield curve had a relatively small linear Ridge coefficient.

Its stronger tree-based importance may suggest that the relationship is nonlinear or regime-dependent.

For example, the feature may only become useful under particular combinations of market conditions.

However, split importance alone is not sufficient to conclude that the yield curve has a strong economic effect.

Later SHAP analysis can provide a more detailed interpretation.

---

# Feature Importance Stability

The importance of several features changes noticeably across test years.

For example:

```text
VIX
volatility_20d
volatility_10d
vix_vs_realized_vol
Treasury variables
```

do not receive exactly the same importance across every fold.

This means the nonlinear tree structure changes as new historical observations are added to the training sample.

This is not necessarily a weakness.

It reflects the greater flexibility of LightGBM.

However, it also suggests that some learned relationships may be more regime-dependent than those captured by Ridge.

---

# Forecast vs Realised Volatility

The LightGBM out-of-sample prediction series generally follows broad movements in realised volatility.

When volatility increases, predictions usually increase.

When volatility falls, predictions generally decline.

The model therefore captures changes in volatility regimes.

However, predictions remain smoother than the realised target during many periods.

---

# Behaviour During Extreme Events

The 2025 volatility episode reveals an important difference between Ridge and LightGBM.

The realised volatility peak reaches approximately:

```text
1.9
```

LightGBM reaches approximately:

```text
1.7
```

while Ridge predicts a substantially lower value during part of the same stress episode.

This indicates that LightGBM can react more aggressively to nonlinear stress signals.

However, better capture of one extreme peak does not imply better overall forecasting accuracy.

The largest-error analysis shows why.

---

# Largest Forecast Errors

The largest LightGBM errors are strongly concentrated around abrupt volatility-regime transitions, particularly March and April 2025.

## Initial Underprediction

A clear example occurs on 2 April 2025:

```text
True forward volatility:      1.9295
LightGBM prediction:          0.3899
Absolute error:               1.5396

VIX:                          21.51
5-day realised volatility:    0.1753
20-day realised volatility:   0.1990
5-day return:                -0.0072
```

At the forecast date, the observable indicators were still relatively moderate.

However, realised volatility over the following five trading days increased dramatically.

The model therefore failed to anticipate the beginning of the new volatility regime.

This illustrates a fundamental difficulty of the forecasting problem:

> an unexpected shock may occur before current market indicators fully reflect the new regime.

---

# Delayed Overreaction

A few days later, the opposite pattern appears.

On 10 April 2025:

```text
True forward volatility:      0.4769
LightGBM prediction:          1.7231
Absolute error:               1.2462

VIX:                          40.72
5-day realised volatility:    0.9201
20-day realised volatility:   0.4825
5-day return:                -0.0241
```

By this point, the market indicators clearly show extreme stress.

LightGBM reacts strongly and predicts continued extreme future volatility.

However, subsequently realised volatility has already begun to decline.

The model therefore significantly overpredicts future risk.

---

# Regime-Transition Pattern

The 2025 errors reveal the following pattern:

```text
Before the shock
    ↓
observable indicators remain relatively moderate
    ↓
future volatility is underestimated


Shock becomes visible
    ↓
VIX and recent realised volatility increase
    ↓
LightGBM reacts aggressively


After the shock
    ↓
stress indicators remain elevated
    ↓
future volatility begins falling
    ↓
LightGBM can remain too high
```

The model therefore appears to enter and exit some extreme volatility regimes with a delay.

---

# Similar Errors in 2022

Several large LightGBM errors also occur during 2022.

For example:

```text
2022-06-07
True volatility      ≈ 0.89
Prediction           ≈ 0.38
```

while other observations show the opposite pattern:

```text
2022-07-05
True volatility      ≈ 0.34
Prediction           ≈ 0.85
```

This suggests that the regime-transition issue is not unique to the 2025 stress episode.

The model can both underreact before a volatility increase and overreact after volatility has already begun to normalise.

---

# Why RMSE Is High in 2025

The 2025 LightGBM results are:

```text
MAE  = 0.1433
RMSE = 0.2536
```

The large difference between MAE and RMSE is explained by a small number of extremely large errors.

Examples include absolute errors of approximately:

```text
1.54
1.25
1.16
1.15
1.00
0.97
```

Because RMSE squares prediction errors, these observations contribute disproportionately to the final metric.

LightGBM therefore does not perform badly on every 2025 observation.

Instead, a relatively small number of regime-transition errors have a very large impact on RMSE.

---

# Ridge vs LightGBM Behaviour

The direct forecast comparison reveals a useful trade-off.

## Ridge

Ridge produces relatively smooth predictions.

It tends to:

```text
react less aggressively
```

to sudden changes.

This can lead to:

```text
underprediction of extreme peaks
```

but also prevents some large overshooting errors.

---

## LightGBM

LightGBM is more reactive.

It can capture some extreme volatility increases more closely than Ridge.

However, the same flexibility can produce:

```text
large overpredictions
```

when market conditions normalise quickly after a shock.

The trade-off can therefore be summarised as:

```text
Ridge
→ smoother
→ more stable
→ lower overall MAE and RMSE
→ may underestimate some extreme peaks


LightGBM
→ more reactive
→ captures some nonlinear stress events better
→ but can overshoot
→ produces larger errors around some regime transitions
```

---

# Why Ridge May Outperform LightGBM Overall

Several factors may explain Ridge's stronger pooled performance.

## Small Dataset

The training folds contain roughly:

```text
1300 to 2100 observations
```

with only 14 features.

This is a relatively small tabular dataset.

A regularised linear model can generalise very effectively in this setting.

---

## Strong Linear Signal

The exploratory analysis showed strong relationships between the target and:

```text
VIX
volatility_5d
volatility_10d
volatility_20d
recent returns
```

A substantial part of the available signal may therefore already be captured through linear relationships.

---

## Greater Flexibility Is Not Always Better

LightGBM has greater modelling flexibility.

This helps capture nonlinear patterns but also makes the fitted structure more sensitive to the historical sample.

The year-specific importance analysis suggests that the tree structure changes more substantially across folds than the core Ridge relationships.

This flexibility can improve some regimes while reducing generalisation in others.

---

## Untuned Hyperparameters

The LightGBM configuration used here is intentionally not heavily tuned.

Therefore, the current conclusion should be:

> this initial LightGBM specification does not outperform Ridge.

It should not be interpreted as evidence that every possible LightGBM configuration would perform worse.

However, hyperparameter tuning should not be used simply to force the nonlinear model to outperform Ridge.

Any tuning must remain leakage-safe and use training data only.

---

# Current Model Ranking

Based on pooled out-of-sample performance:

| Rank | Model | MAE | RMSE |
|---:|---|---:|---:|
| 1 | Ridge | **0.1222** | **0.1772** |
| 2 | LightGBM | 0.1286 | 0.1877 |
| 3 | Persistence | 0.1982 | 0.2574 |

Ridge is therefore the current best model.

---

# What Each Model Teaches

The three models provide different insights.

## Persistence

Persistence demonstrates that:

```text
recent volatility
```

contains useful information about future volatility.

However, it performs poorly during abrupt changes in market conditions.

---

## Ridge

Ridge demonstrates that combining:

```text
VIX
multiple realised-volatility horizons
recent equity returns
VIX changes
rate variables
```

substantially improves forecasting performance.

The model is stable and achieves the best overall out-of-sample accuracy.

---

## LightGBM

LightGBM demonstrates that nonlinear relationships and feature interactions can improve performance in some regimes.

It outperforms Ridge in 2022 and captures some large volatility movements more aggressively.

However, it also makes larger errors during some abrupt regime transitions.

---

# Main Model-Comparison Conclusion

The experiment shows that **more complex models do not automatically produce better financial forecasts**.

The best-performing current model is Ridge regression.

Its pooled out-of-sample performance is:

```text
MAE  = 0.1222
RMSE = 0.1772
```

compared with:

```text
LightGBM
MAE  = 0.1286
RMSE = 0.1877
```

and:

```text
Persistence
MAE  = 0.1982
RMSE = 0.2574
```

Ridge therefore provides the strongest balance between predictive accuracy, stability, and interpretability.

LightGBM remains valuable because it reveals nonlinear behaviour and responds more aggressively to some extreme volatility episodes.

The model comparison suggests that the forecasting problem is not simply about choosing the most sophisticated algorithm.

Instead, performance depends on:

```text
signal strength
model complexity
market regime
feature stability
extreme-event behaviour
```

---

# Current Research Answer

The project initially asked:

> Can machine learning improve one-week-ahead realised-volatility forecasts relative to simple volatility persistence?

The current answer is:

> Yes. Both Ridge and LightGBM substantially outperform the persistence baseline across the 2022–2025 walk-forward evaluation.

A second question was:

> Does nonlinear modelling improve further on a regularised linear model?

The current answer is:

> Not overall. LightGBM improves performance in some individual periods, but Ridge achieves lower pooled MAE and RMSE and remains the strongest overall model.

---

# Next Steps

The next stage should evaluate model behaviour specifically across volatility regimes.

The objective is to determine whether the overall Ridge advantage also holds during:

```text
normal volatility
elevated volatility
stress periods
```

This is important because average MAE and RMSE can hide poor performance during the periods that matter most for financial-risk applications.

The next analysis will therefore compare Persistence, Ridge, and LightGBM separately across market-volatility regimes.