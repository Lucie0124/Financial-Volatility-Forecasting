# Results

## Overview

This document summarizes the main out-of-sample results of the volatility-forecasting experiment.

Three forecasting approaches are compared:

1. **Persistence baseline**
2. **Ridge regression**
3. **LightGBM regression**

The forecasting objective is to predict **five-trading-day forward realised S&P 500 volatility** using information available at the end of day \(t\).

All reported model results use the same leakage-safe expanding walk-forward evaluation:

```text
Train: historical observations before test year
Test:  one full calendar year

Test years:
2022
2023
2024
2025
```

A five-observation purge is applied at every train/test boundary because each target uses returns from the following five trading days.

The primary evaluation metric is:

```text
MAE
```

with:

```text
RMSE
```

used as a secondary metric to place greater weight on extreme forecast errors.

---

# Models

## Persistence

The persistence forecast assumes that recent volatility persists into the future:

```text
forecast
=
current 5-day realised volatility
```

This provides a simple but meaningful benchmark.

---

## Ridge Regression

Ridge combines all modelling features through a regularised linear model.

The main input groups include:

```text
VIX
recent S&P 500 returns
5-day realised volatility
10-day realised volatility
20-day realised volatility
VIX changes
VIX vs realised-volatility spread
Treasury yields
yield curve
Treasury yield changes
```

Features are standardised using training data only.

---

## LightGBM

LightGBM provides a nonlinear tree-based alternative.

It can represent:

```text
threshold effects
nonlinear relationships
feature interactions
```

without requiring feature standardisation.

The initial LightGBM configuration is deliberately conservative and is not heavily hyperparameter-tuned.

---

# Overall Out-of-Sample Performance

The pooled walk-forward results are:

| Model | Pooled MAE | Pooled RMSE |
|---|---:|---:|
| Persistence | 0.1982 | 0.2574 |
| Ridge | **0.1222** | **0.1772** |
| LightGBM | 0.1286 | 0.1877 |

The overall ranking is:

```text
1. Ridge
2. LightGBM
3. Persistence
```

Ridge achieves the lowest pooled MAE and RMSE.

---

# Improvement Relative to Persistence

Ridge reduces pooled MAE from:

```text
0.1982
```

to:

```text
0.1222
```

which corresponds to approximately:

```text
38% lower MAE
```

Ridge also reduces pooled RMSE by approximately:

```text
31%
```

LightGBM reduces pooled MAE to:

```text
0.1286
```

which corresponds to approximately:

```text
35% lower MAE
```

relative to persistence.

Its pooled RMSE is approximately:

```text
27% lower
```

than persistence.

Both fitted models therefore substantially outperform the simple persistence benchmark.

---

# Year-by-Year Performance

## MAE

| Test Year | Persistence | Ridge | LightGBM |
|---|---:|---:|---:|
| 2022 | 0.2943 | 0.1779 | **0.1602** |
| 2023 | 0.1565 | **0.1107** | 0.1241 |
| 2024 | 0.1509 | **0.0845** | 0.0872 |
| 2025 | 0.1911 | **0.1158** | 0.1433 |

The ranking changes across years.

LightGBM performs best in 2022, while Ridge performs best in 2023, 2024, and 2025.

This demonstrates that no model dominates in every market environment.

---

# Interpretation of Overall Results

The nonlinear LightGBM model does not automatically outperform the simpler Ridge model.

This is an important result.

The current dataset contains approximately:

```text
1300 to 2100 training observations per fold
```

with only 14 modelling features.

Several predictors, particularly:

```text
VIX
recent realised volatility
recent returns
```

have strong and relatively stable relationships with future volatility.

A regularised linear model is therefore able to capture a large proportion of the available predictive signal.

The results suggest that additional nonlinear flexibility can be useful in some market regimes, but does not improve average generalisation sufficiently to outperform Ridge overall.

---

# Ridge Behaviour

Ridge produces relatively smooth forecasts.

It generally captures broad changes in the volatility environment but compresses the range of predictions.

The typical pattern is:

```text
very low realised volatility
→ Ridge can overpredict

moderate volatility
→ Ridge tracks reasonably well

extreme volatility
→ Ridge tends to underpredict
```

This behaviour is consistent with regression toward moderate values.

Ridge's strength is therefore not perfect tracking of extreme events, but relatively stable performance across the full out-of-sample period.

---

# LightGBM Behaviour

LightGBM reacts more aggressively to changing market conditions.

Its nonlinear structure allows the model to respond more strongly when combinations of features indicate stress.

This helps explain why LightGBM outperforms Ridge in 2022 and captures some sharp volatility increases more strongly.

However, this flexibility can also generate larger forecast errors.

In particular, LightGBM can remain at extremely high forecast levels after the realised-volatility regime has already started to normalise.

This creates a trade-off:

```text
Ridge
→ smoother
→ more stable
→ better aggregate accuracy

LightGBM
→ more reactive
→ better in some stress episodes
→ larger errors during some transitions
```

---

# Largest LightGBM Errors

The largest LightGBM errors are strongly concentrated around abrupt regime changes.

One important example is:

```text
2 April 2025
```

with:

```text
True forward volatility      = 1.9295
LightGBM prediction          = 0.3899
Absolute error               = 1.5396
```

At the forecast date:

```text
VIX                         = 21.51
5-day realised volatility   = 0.1753
20-day realised volatility  = 0.1990
```

The available market indicators were still relatively moderate.

The realised volatility over the following five days then increased dramatically.

This demonstrates the difficulty of predicting an abrupt shock before current indicators have fully reacted.

---

A few days later, the opposite problem appears.

On:

```text
10 April 2025
```

LightGBM produces:

```text
True forward volatility      = 0.4769
Prediction                   = 1.7231
Absolute error               = 1.2462
```

At that point:

```text
VIX                         = 40.72
5-day realised volatility   = 0.9201
20-day realised volatility  = 0.4825
```

Current indicators now clearly reflect extreme stress.

LightGBM therefore predicts continued extreme future volatility.

However, forward realised volatility has already started to decline.

The model therefore exhibits an important regime-transition pattern:

```text
Before the shock
    ↓
current features still look relatively moderate
    ↓
future stress is underestimated

During the shock
    ↓
market indicators increase sharply
    ↓
LightGBM reacts aggressively

After the shock
    ↓
current stress indicators remain elevated
    ↓
future volatility begins falling
    ↓
LightGBM can overpredict
```

This suggests that many available predictors are effective at describing the **current state** of market stress but may react with delay when the underlying regime changes abruptly.

---

# Feature-Level Findings

Across the exploratory analysis and fitted models, the strongest recurring predictors are primarily:

```text
VIX
recent realised volatility
recent equity returns
```

VIX is particularly consistent.

It shows:

```text
strong correlation with future realised volatility
large positive Ridge coefficient
high LightGBM split importance
```

Recent realised-volatility measures are also important across both model families.

This supports the persistence of volatility through time.

Recent equity returns generally receive negative Ridge coefficients:

```text
market declines
→ higher predicted future volatility
```

Treasury and yield-curve variables provide weaker and less stable linear signals.

However, some of these variables receive meaningful LightGBM split importance, suggesting that their information may be nonlinear or regime-dependent.

---

# Volatility-Regime Evaluation

Overall MAE and RMSE can hide important differences between calm and stressed market environments.

The out-of-sample observations are therefore divided into three realised-volatility regimes:

```text
Normal
Elevated
Stress
```

For each walk-forward fold, regime thresholds are calculated from the training target distribution only.

The thresholds are:

```text
Normal
target < training 70th percentile

Elevated
training 70th percentile <= target < training 90th percentile

Stress
target >= training 90th percentile
```

This avoids using future test-period information when determining historical regime thresholds.

---

# Regime Thresholds

The fold-specific thresholds are:

| Test Year | Elevated Threshold | Stress Threshold |
|---|---:|---:|
| 2022 | 0.3089 | 0.5549 |
| 2023 | 0.3821 | 0.6263 |
| 2024 | 0.3726 | 0.5924 |
| 2025 | 0.3583 | 0.5731 |

The changing thresholds show that the historical volatility distribution evolves over time.

A fixed threshold over the full dataset would therefore provide a less realistic evaluation.

---

# Regime Distribution

Across the 2022–2025 out-of-sample predictions:

| Regime | Observations |
|---|---:|
| Normal | 637 |
| Elevated | 259 |
| Stress | 107 |

Normal conditions therefore represent the majority of test observations.

This is important when interpreting overall metrics.

Because approximately two thirds of observations are classified as Normal, performance in this regime has a large influence on pooled MAE and RMSE.

---

# MAE by Volatility Regime

The regime-specific MAE results are:

| Model | Normal | Elevated | Stress |
|---|---:|---:|---:|
| Persistence | 0.1212 | 0.2611 | 0.5043 |
| Ridge | **0.0933** | 0.1363 | 0.2600 |
| LightGBM | 0.1073 | **0.1307** | **0.2511** |

The relative model ranking changes with volatility conditions.

---

## Normal Regime

Ridge performs best:

```text
Ridge MAE      = 0.0933
LightGBM MAE   = 0.1073
Persistence    = 0.1212
```

This helps explain why Ridge is the strongest model overall.

Most observations occur during Normal conditions, where Ridge has a clear advantage.

The linear model therefore provides strong generalisation during ordinary market environments.

---

## Elevated Regime

LightGBM slightly outperforms Ridge:

```text
LightGBM MAE = 0.1307
Ridge MAE    = 0.1363
```

The difference is modest, but it suggests that nonlinear effects become somewhat more useful when volatility rises above normal levels.

Both models perform dramatically better than persistence:

```text
Persistence MAE = 0.2611
```

---

## Stress Regime

LightGBM also achieves the lowest MAE during Stress periods:

```text
LightGBM MAE = 0.2511
Ridge MAE    = 0.2600
```

The difference is relatively small, but the result supports the idea that LightGBM's nonlinear structure is useful when the market moves into extreme conditions.

Persistence performs poorly:

```text
Persistence MAE = 0.5043
```

The fitted models therefore reduce the average absolute stress-period forecast error by roughly half relative to persistence.

---

# RMSE by Volatility Regime

The regime-specific RMSE results are:

| Model | Normal | Elevated | Stress |
|---|---:|---:|---:|
| Persistence | 0.1388 | 0.2770 | 0.5663 |
| Ridge | **0.1258** | **0.1722** | **0.3581** |
| LightGBM | 0.1370 | 0.1885 | 0.3643 |

Ridge achieves the lowest RMSE in all three regimes.

This creates an important distinction between MAE and RMSE.

In Elevated and Stress regimes:

```text
LightGBM
→ slightly lower MAE

Ridge
→ slightly lower RMSE
```

This means LightGBM tends to have somewhat smaller typical errors in these regimes but also produces a few larger mistakes.

Because RMSE squares errors, those extreme misses increase LightGBM's RMSE.

This is consistent with the large-error analysis around abrupt 2025 regime transitions.

---

# Forecast Underestimation Rate

For risk forecasting, error direction is also important.

The underestimation rate measures:

```text
fraction of observations where

actual realised volatility
>
forecast volatility
```

A high value means the model frequently forecasts less risk than subsequently materialises.

The results are:

| Model | Normal | Elevated | Stress |
|---|---:|---:|---:|
| Persistence | 90.9% | 98.8% | 97.2% |
| Ridge | 23.9% | 61.4% | 75.7% |
| LightGBM | **18.5%** | **45.9%** | **61.7%** |

This metric reveals a substantial difference between Ridge and LightGBM.

---

# Persistence Underestimation

Persistence underestimates future volatility in nearly every Elevated and Stress observation:

```text
Elevated ≈ 99%
Stress   ≈ 97%
```

This demonstrates an important weakness of a backward-looking persistence forecast during periods where future volatility increases substantially.

However, this result should be interpreted carefully.

Regimes are defined using the **future realised target**.

Stress observations are therefore specifically observations where future volatility turned out high.

A backward-looking persistence feature will naturally tend to lag those increases.

The underestimation pattern should therefore not be interpreted as a universal property of persistence under every possible regime definition.

---

# Ridge Underestimation

Ridge dramatically improves on persistence.

In Normal conditions it underestimates future volatility only:

```text
23.9%
```

of the time.

However, the rate rises as realised volatility increases:

```text
Normal      23.9%
Elevated    61.4%
Stress      75.7%
```

Ridge therefore becomes increasingly biased toward underestimating future volatility during high-risk periods.

This is consistent with its smoother predictions and tendency to underestimate extreme peaks.

---

# LightGBM Underestimation

LightGBM has the lowest underestimation rate in every regime:

```text
Normal      18.5%
Elevated    45.9%
Stress      61.7%
```

The difference is particularly meaningful during Stress periods.

Compared with Ridge:

```text
Ridge stress underestimation      = 75.7%
LightGBM stress underestimation   = 61.7%
```

LightGBM is therefore less systematically biased toward predicting too little volatility during extreme market conditions.

From a financial-risk perspective, this is an attractive property.

Underestimating volatility during stress can be more consequential than producing a slightly conservative forecast.

---

# Main Regime Trade-Off

The regime analysis reveals that there is no universally superior model.

The comparison is:

```text
Normal regime
→ Ridge has best MAE and RMSE

Elevated regime
→ LightGBM has best MAE
→ Ridge has best RMSE

Stress regime
→ LightGBM has best MAE
→ Ridge has best RMSE
```

LightGBM also has the lowest underestimation rate in every regime.

This produces a clear model trade-off.

---

## Ridge

Strengths:

```text
best overall MAE
best overall RMSE
best Normal-regime MAE
best RMSE in every regime
stable predictions
strong interpretability
```

Weaknesses:

```text
smoother response to shocks
underestimates many Stress observations
can miss the magnitude of sudden volatility spikes
```

---

## LightGBM

Strengths:

```text
best Elevated-regime MAE
best Stress-regime MAE
lowest underestimation rate
more reactive during stress
captures nonlinear relationships
```

Weaknesses:

```text
slightly worse pooled performance
larger extreme errors
can overshoot after volatility shocks
less directly interpretable
```

---

# Final Regression Model Conclusion

For general-purpose five-day volatility forecasting, **Ridge is currently the strongest model**.

It provides the best aggregate out-of-sample performance:

```text
Pooled MAE  = 0.1222
Pooled RMSE = 0.1772
```

It also achieves the lowest RMSE across all three volatility regimes.

Ridge therefore provides the strongest overall balance of:

```text
accuracy
stability
interpretability
```

---

However, LightGBM provides valuable complementary behaviour.

It achieves:

```text
lower MAE in Elevated volatility
lower MAE in Stress volatility
lower underestimation rate in every regime
```

This suggests that nonlinear modelling becomes more useful when market conditions move away from normality.

The final interpretation is therefore not simply:

```text
Ridge good
LightGBM bad
```

Instead:

```text
Ridge
→ strongest general-purpose forecast model

LightGBM
→ more reactive stress-oriented model
```

---

# Main Research Conclusions

## 1. Machine learning improves materially on volatility persistence

Both Ridge and LightGBM substantially outperform the persistence baseline.

This shows that future volatility contains useful predictive information beyond recent realised volatility alone.

---

## 2. Greater model complexity does not guarantee better overall performance

LightGBM is more flexible than Ridge but does not achieve lower pooled MAE or RMSE.

This demonstrates the importance of evaluating models out of sample rather than assuming that a more sophisticated algorithm will perform better.

---

## 3. VIX and recent realised volatility provide the strongest recurring signals

Across:

```text
EDA
Ridge coefficients
LightGBM importance
```

VIX and recent realised-volatility features consistently emerge as important predictors.

---

## 4. Model performance depends strongly on the volatility regime

Ridge performs best in Normal conditions, while LightGBM gains an MAE advantage in Elevated and Stress periods.

Aggregate metrics therefore hide meaningful regime-dependent behaviour.

---

## 5. Stress forecasting remains difficult

All models have substantially larger errors during Stress periods.

For example:

```text
Ridge MAE

Normal = 0.0933
Stress = 0.2600
```

and:

```text
LightGBM MAE

Normal = 0.1073
Stress = 0.2511
```

Extreme volatility is therefore much harder to forecast than ordinary conditions.

---

## 6. Sudden regime transitions are the main failure mode

The largest errors occur around abrupt transitions where:

```text
current market conditions
```

and:

```text
future realised volatility
```

change rapidly relative to one another.

A model may:

```text
underpredict before a shock
```

and then:

```text
overpredict after the shock
```

because observable stress indicators themselves react with delay.

---

## 7. LightGBM reduces systematic stress underestimation

During Stress observations:

```text
Persistence underestimation = 97.2%
Ridge underestimation       = 75.7%
LightGBM underestimation    = 61.7%
```

LightGBM therefore appears more responsive to high-risk conditions even though Ridge remains slightly better on stress RMSE.

---

# Current Answer to the Forecasting Question

The project asks:

> Can historical market information improve five-day-ahead realised-volatility forecasts relative to simple volatility persistence?

The current evidence suggests:

> Yes. Both Ridge regression and LightGBM substantially improve out-of-sample forecasting accuracy relative to persistence.

A second question is:

> Does nonlinear modelling provide additional value beyond a regularised linear model?

The answer is more nuanced:

> Not overall. Ridge achieves the best pooled MAE and RMSE. However, LightGBM achieves lower MAE during Elevated and Stress regimes and substantially reduces the frequency of volatility underestimation during Stress periods.

---

# Final Regression Model Ranking

For overall forecasting:

```text
1. Ridge
2. LightGBM
3. Persistence
```

For stress-oriented behaviour:

```text
MAE
1. LightGBM
2. Ridge
3. Persistence

RMSE
1. Ridge
2. LightGBM
3. Persistence

Underestimation rate
1. LightGBM
2. Ridge
3. Persistence
```

The results therefore support keeping both Ridge and LightGBM in the project rather than treating one as redundant.

---

# Next Step

The regression models predict continuous future volatility.

The next stage extends the project to a related classification problem:

> Can current market features predict whether the next five trading days will belong to a Normal, Elevated, or Stress volatility regime?

This differs from the current regime analysis.

The current regime labels are used **after the fact** to evaluate regression forecasts.

The next model will instead attempt to predict the future regime directly using only information available at forecast time.

This creates a complementary system:

```text
Regression model
→ predicts future volatility level

Classification model
→ predicts future volatility regime
```

The classification stage will therefore focus on:

```text
Normal
Elevated
Stress
```

with particular attention to the ability to detect Stress observations.