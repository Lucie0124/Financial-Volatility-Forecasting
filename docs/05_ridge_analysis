# Ridge Regression: Implementation and Analysis

## Overview

After establishing the persistence baseline, the first fitted machine-learning model is a **Ridge regression**.

The main objective is to answer:

> Can a linear model combining multiple market indicators improve five-day forward realised-volatility forecasts relative to using recent realised volatility alone?

The persistence baseline uses only:

```text
volatility_5d
      ↓
prediction of future 5-day volatility
```

Ridge instead combines information from:

```text
VIX
recent S&P 500 returns
recent realised volatility
VIX changes
Treasury yields
yield-curve information
recent interest-rate changes
```

The model is evaluated using exactly the same chronological walk-forward framework as the persistence benchmark.

---

# Ridge Regression

## Linear Model

Ridge regression is a regularised version of linear regression.

A standard linear regression predicts:

```math
\hat{y}
=
\beta_0
+
\beta_1x_1
+
\beta_2x_2
+
\cdots
+
\beta_px_p
```

where:

- \(\hat{y}\) is predicted five-day forward realised volatility;
- \(x_j\) represents one model feature;
- \(\beta_j\) represents the coefficient associated with that feature;
- \(\beta_0\) is the intercept.

The model therefore assumes that the forecasting target can be approximated through a linear combination of the input features.

---

## Why Ridge Instead of Ordinary Linear Regression?

Several features contain related information.

For example:

```text
volatility_5d
volatility_10d
volatility_20d
```

all measure recent realised volatility over overlapping time horizons.

Similarly, VIX is also strongly related to current market-volatility conditions.

This creates **multicollinearity**, meaning that several predictors are correlated with one another.

With ordinary linear regression, multicollinearity can make individual coefficients unstable.

Ridge regression addresses this by adding an L2 regularisation penalty.

The objective becomes:

```math
\text{Loss}
=
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
+
\alpha
\sum_{j=1}^{p}
\beta_j^2
```

The first term measures forecasting error.

The second term penalises large coefficients.

The regularisation parameter \(\alpha\) controls the strength of this penalty.

For the initial Ridge benchmark:

```text
alpha = 1.0
```

This value is used as an initial modelling choice rather than an optimised hyperparameter.

Hyperparameter tuning can later be performed using training data only.

---

# Feature Standardisation

## Why Scaling Is Required

The model features use very different numerical scales.

For example:

```text
VIX              -> values often between roughly 10 and 80
Treasury yields  -> values of a few percentage points
daily returns    -> small decimal values
realised vol.    -> decimal annualised volatility
```

Because Ridge penalises coefficient magnitude, using features on very different scales would make coefficient regularisation difficult to interpret consistently.

The features are therefore standardised before fitting Ridge.

For each feature:

```math
z =
\frac{x-\mu}{\sigma}
```

where:

- \(x\) is the original feature value;
- \(\mu\) is the training-set mean;
- \(\sigma\) is the training-set standard deviation.

After transformation, each feature is approximately centred around zero with unit variance.

---

## Preventing Scaling Leakage

The scaler must learn its parameters using **training data only**.

For example, for the 2022 fold:

```text
2016 ───────────── 2021 | 2022
         TRAIN          TEST
```

The feature means and standard deviations are estimated only from the training period.

The 2022 test observations are then transformed using these historical values.

The scaler must not learn the mean or standard deviation of the complete 2022 test period because this would introduce future information into the modelling pipeline.

---

# Scikit-Learn Pipeline

The model is implemented using:

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]
)
```

The pipeline represents the following sequence:

```text
Raw features
     ↓
StandardScaler
     ↓
Standardised features
     ↓
Ridge regression
     ↓
Volatility prediction
```

During training:

```text
X_train
   ↓
fit StandardScaler on X_train
   ↓
transform X_train
   ↓
fit Ridge using X_train and y_train
```

During prediction:

```text
X_test
   ↓
transform using the TRAINING scaler
   ↓
trained Ridge model
   ↓
y_pred
```

Using a pipeline ensures that preprocessing and modelling are applied consistently and reduces the risk of accidentally fitting preprocessing operations on future test data.

---

# Walk-Forward Ridge Evaluation

The Ridge model uses the same expanding-window evaluation procedure as the persistence baseline.

The folds are approximately:

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

The final five training observations are purged before each test period because the five-day forward target could otherwise contain returns belonging to the test period.

For each fold, the pipeline is fitted from scratch.

This means that four different Ridge models are estimated:

```text
2022 model
-> trained using information available before 2022

2023 model
-> retrained using additional history

2024 model
-> retrained again

2025 model
-> retrained again
```

This reflects how a forecasting system could be periodically retrained as new observations become available.

---

# Ridge Results

## Fold-Level Performance

The Ridge walk-forward results are:

| Test Year | Train Size | Test Size | MAE | RMSE |
|---|---:|---:|---:|---:|
| 2022 | 1336 | 251 | 0.1779 | 0.2318 |
| 2023 | 1587 | 250 | 0.1107 | 0.1417 |
| 2024 | 1837 | 252 | 0.0845 | 0.1135 |
| 2025 | 2089 | 250 | 0.1158 | 0.1972 |

The average fold performance is:

```text
Mean fold MAE  = 0.1222
Mean fold RMSE = 0.1711
```

The pooled out-of-sample results are:

```text
Pooled MAE  = 0.1222
Pooled RMSE = 0.1772
```

---

# Comparison with Persistence

The pooled persistence results were:

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence | 0.1982 | 0.2574 |
| Ridge | **0.1222** | **0.1772** |

Ridge therefore reduces pooled MAE by approximately:

```text
38%
```

and pooled RMSE by approximately:

```text
31%
```

relative to persistence.

This is a substantial improvement.

It indicates that future volatility contains useful information beyond the most recent five-day realised-volatility value.

---

## Performance by Year

The MAE comparison is:

| Test Year | Persistence MAE | Ridge MAE | Approx. Improvement |
|---|---:|---:|---:|
| 2022 | 0.2943 | 0.1779 | 39% |
| 2023 | 0.1565 | 0.1107 | 29% |
| 2024 | 0.1509 | 0.0845 | 44% |
| 2025 | 0.1911 | 0.1158 | 39% |

Ridge improves MAE in every walk-forward fold.

This is important because the improvement is not produced by a single unusually favourable test year.

Instead, the additional features provide useful information across several different market periods.

---

# Ridge Coefficient Analysis

Because all features are standardised before fitting Ridge, their coefficient magnitudes can be compared more meaningfully.

A positive coefficient means:

> higher values of the feature are associated with higher predicted future volatility, holding the other model features fixed.

A negative coefficient means:

> higher values of the feature are associated with lower predicted future volatility, holding the other model features fixed.

These coefficients describe relationships learned by the predictive model.

They should **not be interpreted as causal effects**.

---

## Average Coefficients Across Walk-Forward Folds

The largest positive average coefficients are associated with:

```text
volatility_10d
vix
volatility_20d
vix_change_5d
vix_vs_realized_vol
```

The model therefore relies most strongly on two broad categories of information:

```text
Recent realised volatility
+
Option-implied volatility
```

This is consistent with the exploratory analysis, which identified VIX and recent realised-volatility measures as the variables most strongly associated with the forecasting target.

---

## Recent Realised Volatility

`volatility_10d` has the largest average positive coefficient.

`volatility_20d` also receives a substantial positive coefficient.

This indicates that persistent market-volatility conditions are important for predicting volatility over the following week.

Interestingly, the average coefficient of:

```text
volatility_5d
```

is close to zero.

This should **not** be interpreted as evidence that five-day volatility is useless.

The volatility features are strongly correlated:

```text
volatility_5d
volatility_10d
volatility_20d
```

Ridge regularisation can distribute predictive weight between correlated variables.

Once VIX and longer volatility horizons are included, `volatility_5d` may contain relatively little additional independent linear information.

This illustrates why individual coefficients must be interpreted carefully when predictors are correlated.

---

# VIX-Related Features

## VIX Level

VIX has one of the largest and most consistently positive coefficients.

The relationship can be summarised as:

```text
higher VIX
    ↓
higher predicted future realised volatility
```

This agrees with both the correlation analysis and the VIX-versus-realised-volatility scatter plot from the exploratory analysis.

---

## Five-Day VIX Change

`vix_change_5d` also receives a positive coefficient.

This means the model uses not only the current level of VIX, but also information about whether implied volatility has recently increased.

A rising VIX over several trading days may therefore provide additional information about changing market conditions.

---

## VIX vs Recent Realised Volatility

`vix_vs_realized_vol` also has a consistently positive coefficient.

This feature measures the difference between:

```text
option-implied volatility
and
recently realised volatility
```

A larger positive difference indicates that option-market implied volatility is elevated relative to recent realised conditions.

The positive Ridge coefficient suggests that this difference contains additional linear information about future realised volatility.

---

# Equity Return Features

Both:

```text
return_1d
return_5d
```

have negative average coefficients.

This means:

```text
recent S&P 500 return decreases
              ↓
predicted future volatility increases
```

Recent market declines are therefore associated with higher subsequent volatility in the fitted linear model.

The five-day return displays a stronger negative relationship than the one-day return.

This result is consistent with the negative correlations observed during exploratory analysis.

---

# Treasury and Yield-Curve Features

Treasury-related features generally receive smaller coefficients than VIX and realised-volatility variables.

Their coefficients also appear less stable across walk-forward folds.

For example:

```text
treasury_2y_change_5d
```

has a strongly negative coefficient in the earliest fold but becomes substantially smaller in later folds.

Similarly, some other rate-related coefficients move closer to zero or change sign as the training sample expands.

This suggests that the relationship between short-term equity volatility and interest-rate variables may be more regime-dependent.

These variables may still contain useful information, but their linear effects appear less stable than those of VIX and recent realised volatility.

---

# Coefficient Stability Across Walk-Forward Folds

The year-by-year coefficient graph provides information that cannot be obtained from average coefficients alone.

Some features have relatively stable coefficient signs and magnitudes across all four models.

Examples include:

```text
volatility_10d
vix
volatility_20d
vix_vs_realized_vol
return_1d
return_5d
```

These features therefore appear to provide relatively consistent linear information as the training sample expands.

Other variables display greater coefficient instability.

For example:

```text
volatility_5d
treasury_10y_change_5d
treasury_2y_change_5d
```

change noticeably across folds.

A coefficient that changes substantially over time may indicate:

- changing market regimes;
- interaction with correlated predictors;
- limited independent predictive information;
- instability in the underlying linear relationship.

This is one reason why both average coefficients and fold-specific coefficients are inspected.

---

# Forecast vs Realised Volatility

The out-of-sample predictions from all four Ridge folds were plotted against the actual five-day forward realised volatility.

The model generally follows broad changes in volatility through time.

When market volatility rises, Ridge predictions usually increase.

When volatility falls, predictions generally decline.

This confirms visually that the model captures useful information about changing volatility regimes.

---

## Predictions Are Smoother Than the Target

The Ridge prediction series is noticeably smoother than the realised-volatility target.

This is an important characteristic of the linear model.

Ridge tends to produce predictions closer to moderate volatility values rather than reproducing the full magnitude of extreme observations.

Conceptually:

```text
Very low realised volatility
        ↓
Ridge may predict slightly too high

Moderate realised volatility
        ↓
Ridge often tracks reasonably well

Extreme realised volatility
        ↓
Ridge tends to predict too low
```

This behaviour is consistent with regression toward the centre of the target distribution.

---

# Extreme Volatility Events

The largest discrepancy visible in the forecast plot occurs around the sharp volatility increase in 2025.

Realised annualised volatility rises to approximately:

```text
1.9
```

while the Ridge prediction peaks substantially below this value.

The model correctly identifies that market conditions have become unusually volatile, but it underestimates the **magnitude** of the extreme event.

This explains why 2025 has:

```text
MAE  = 0.1158
RMSE = 0.1972
```

The average forecasting error remains relatively low, but a limited number of large errors increase RMSE substantially.

This indicates that Ridge improves general forecasting accuracy while still struggling with abrupt tail events.

---

# Calm-Period Forecasting

The plot also shows some periods, particularly during 2023 and 2024, where realised volatility falls to very low levels while Ridge remains somewhat higher.

The model can therefore:

```text
underestimate extreme high-volatility periods
```

while also:

```text
overestimate exceptionally calm periods
```

This reinforces the conclusion that linear Ridge predictions are smoother than the realised target.

---

# What Ridge Adds Beyond Persistence

The persistence model uses a single source of information:

```text
recent 5-day realised volatility
```

Ridge improves upon this by combining several complementary signals.

The coefficient analysis suggests that much of the additional information comes from:

```text
VIX level
+
medium-term realised volatility
+
changes in VIX
+
VIX relative to recent realised volatility
+
recent equity-market returns
```

This provides a plausible explanation for the substantial reduction in out-of-sample forecasting error.

---

# Limitations of Ridge

Despite its strong improvement over persistence, Ridge has several limitations.

## Linear Relationships

Ridge assumes that features contribute linearly to the prediction.

For example, it effectively assumes that a one-standard-deviation VIX increase has a similar marginal linear effect regardless of whether VIX is:

```text
15
30
or
60
```

Actual financial relationships may instead contain thresholds or nonlinear behaviour.

---

## Feature Interactions

Ridge does not naturally capture complex interactions.

For example, the effect of an increasing VIX may depend on whether:

```text
recent returns are strongly negative
```

or whether:

```text
recent realised volatility is already elevated
```

These interactions may contain useful information that a simple linear specification cannot represent directly.

---

## Extreme Events

The prediction plot indicates that Ridge tends to underestimate abrupt extreme volatility spikes.

This suggests that the linear model has difficulty representing the tails of the target distribution.

These periods are particularly important for a risk-forecasting application.

---

# Motivation for the Next Model

The next modelling stage uses LightGBM.

The central question becomes:

> Can a nonlinear tree-based model capture threshold effects, feature interactions, and stress-regime behaviour that Ridge cannot?

The progression of the project is therefore:

```text
Persistence
    ↓
Can recent volatility alone forecast future volatility?

Ridge
    ↓
Can multiple market indicators improve the forecast
through linear relationships?

LightGBM
    ↓
Can nonlinear relationships and feature interactions
improve performance further?
```

---

# Main Ridge Conclusions

The Ridge experiment provides several important findings.

### 1. Ridge substantially outperforms persistence

Pooled out-of-sample MAE decreases from:

```text
0.1982 → 0.1222
```

and pooled RMSE decreases from:

```text
0.2574 → 0.1772
```

corresponding to improvements of approximately **38% in MAE** and **31% in RMSE**.

### 2. The improvement is consistent across test years

Ridge achieves lower MAE than persistence in all four walk-forward folds.

The result is therefore not driven by a single test period.

### 3. Volatility and VIX dominate the linear model

The largest and most stable positive coefficients are associated primarily with recent realised volatility and VIX-related variables.

### 4. Recent market declines contain additional information

The negative coefficients on recent S&P 500 returns indicate that declining equity markets are associated with higher predicted subsequent volatility.

### 5. Rate-related relationships are less stable

Treasury and yield-curve features generally contribute less and display greater variation across walk-forward folds.

### 6. Ridge captures volatility regimes but smooths extreme outcomes

The predicted series generally follows broad movements in realised volatility but tends to underestimate sudden volatility spikes and sometimes overestimate very calm periods.

Overall, Ridge demonstrates that combining economically interpretable market indicators provides substantial predictive value beyond simple volatility persistence.

Its remaining weakness around nonlinear and extreme market behaviour provides the motivation for evaluating a nonlinear model next.