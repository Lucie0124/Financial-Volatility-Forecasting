# Modelling Methodology

## Forecasting Problem

The objective of the project is to predict **S&P 500 five-day forward realised volatility** using only information available at the end of each trading day.

The forecasting problem can be written as:

```math
X_t \rightarrow RV^{(5)}_t
```

where:

- \(X_t\) contains market information observed on or before trading day \(t\);
- \(RV^{(5)}_t\) is the realised volatility observed over trading days \(t+1\) to \(t+5\).

The project is therefore a supervised regression problem.

The model does not attempt to predict whether the S&P 500 will rise or fall. Instead, it predicts the **magnitude of future market fluctuations**.

---

## Temporal Structure and Leakage Prevention

Financial time series require strict chronological separation between training data and future observations.

Random train-test splitting is therefore not appropriate for this project.

For example, a random split could produce:

```text
Training data:
2018
2020
2023
2025

Test data:
2019
2021
2024
```

In this situation, the model would effectively be trained using observations that occur after some of the test observations.

This does not represent a realistic forecasting scenario.

Instead, the project always follows:

```text
PAST
  ↓
training data
  ↓
model
  ↓
future unseen data
  ↓
evaluation
```

All features are constructed using information available on or before date \(t\).

Future returns are used exclusively to construct the forecasting target.

Backward-filling is avoided because it could propagate future observations into earlier dates.

---

## Five-Day Forecast Horizon and Target Overlap

The forecasting target uses the following five trading days:

```text
Prediction date t

        future target window
             ↓
t | t+1 t+2 t+3 t+4 t+5
```

Consecutive target observations therefore overlap.

For example:

```text
Target at t:
t+1  t+2  t+3  t+4  t+5

Target at t+1:
     t+2  t+3  t+4  t+5  t+6
```

These two target values share four out of five future returns.

As a result, neighbouring target observations are not statistically independent.

This is expected given the five-day rolling forecasting horizon.

During the final walk-forward evaluation, the chronological split must ensure that training labels do not use returns belonging to the test period.

A small temporal gap will therefore be introduced at the train-test boundary to avoid overlap between the last training targets and the first test observations.

---

# Persistence Baseline

## Motivation

Before training machine-learning models, the project establishes a simple forecasting benchmark.

Volatility exhibits persistence and clustering:

```text
recently calm market
        ↓
often followed by
        ↓
relatively calm market

recently volatile market
        ↓
often followed by
        ↓
continued high volatility
```

A simple persistence forecast therefore assumes that the volatility observed during the most recent five trading days will continue during the following five trading days.

The forecast is:

```math
\hat{RV}^{(5)}_t = Volatility^{(5)}_t
```

where:

- \(\hat{RV}^{(5)}_t\) is the predicted five-day forward realised volatility;
- \(Volatility^{(5)}_t\) is the realised volatility calculated from the most recent five trading days.

In practical terms:

```text
recent 5-day realised volatility
                ↓
        used as prediction
                ↓
future 5-day realised volatility
```

No statistical model needs to be fitted.

The persistence benchmark is important because recent volatility already contains substantial predictive information.

A more complex model should therefore outperform this benchmark in order to demonstrate genuine added value.

---

## Baseline Results

The persistence baseline produces the following results:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.1820 |
| Root Mean Squared Error (RMSE) | 0.2641 |

The target is expressed as annualised volatility in decimal form.

Therefore:

```text
0.182
```

corresponds approximately to:

```text
18.2 annualised volatility percentage points
```

The baseline therefore misses the subsequently realised volatility by approximately **18.2 annualised volatility percentage points on average**.

For example:

```text
True forward volatility:       0.40 = 40%
Predicted forward volatility:  0.22 = 22%

Absolute error:                0.18
                               = 18 volatility percentage points
```

---

# Evaluation Metrics

Two main regression metrics are used to evaluate the forecasts:

1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)

They both measure prediction errors, but they treat large errors differently.

---

## Mean Absolute Error

The Mean Absolute Error is defined as:

```math
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
```

where:

- \(y_i\) is the true forward realised volatility;
- \(\hat{y}_i\) is the model prediction;
- \(N\) is the total number of forecasts.

The MAE calculates the absolute error of each prediction and then averages these errors.

For example:

```text
True volatility:       0.20   0.40   0.30
Predicted volatility:  0.25   0.30   0.35

Error:                 -0.05   0.10  -0.05
Absolute error:         0.05   0.10   0.05
```

The MAE is:

```math
MAE =
\frac{
0.05 + 0.10 + 0.05
}{3}
```

which gives approximately:

```text
MAE = 0.067
```

### Interpretation

MAE is particularly useful because it is expressed in the same units as the forecasting target.

For this project's persistence baseline:

```text
MAE = 0.1820
```

This means that the predicted volatility differs from the true forward realised volatility by approximately:

```text
18.2 annualised volatility percentage points
```

on average.

MAE treats all errors proportionally.

For example:

```text
Error = 0.10
```

counts twice as much as:

```text
Error = 0.05
```

This makes MAE a useful measure of typical forecasting accuracy.

---

## Root Mean Squared Error

The Root Mean Squared Error is defined as:

```math
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
```

RMSE follows three steps:

```text
prediction error
       ↓
square the error
       ↓
average squared errors
       ↓
take the square root
```

The important difference from MAE is that errors are **squared**.

This means large forecasting errors receive much more weight.

For example:

```text
Error = 0.10
Squared error = 0.01
```

while:

```text
Error = 0.50
Squared error = 0.25
```

The second error is five times larger:

```text
0.50 / 0.10 = 5
```

but contributes twenty-five times more squared error:

```text
0.25 / 0.01 = 25
```

RMSE is therefore particularly sensitive to large forecasting mistakes.

---

## Why Use Both MAE and RMSE?

The baseline results are:

```text
MAE  = 0.1820
RMSE = 0.2641
```

The RMSE is noticeably larger than the MAE.

This suggests that the persistence model makes some relatively large forecasting errors.

This is consistent with the exploratory data analysis.

The target distribution is strongly right-skewed and contains a small number of very large volatility spikes.

During relatively stable market periods, the persistence assumption may work reasonably well:

```text
recent volatility ≈ future volatility
```

However, during abrupt regime changes:

```text
recent volatility
      ↓
still relatively moderate

future volatility
      ↓
sudden large spike
```

the persistence model may significantly underestimate future risk.

MAE and RMSE therefore provide complementary information:

```text
MAE
↓
How large is the typical forecasting error?

RMSE
↓
Does the model make particularly large forecasting mistakes?
```

For this reason, both metrics are reported throughout the project.

---

# Interpretation of the Persistence Baseline

The persistence baseline provides a meaningful first benchmark.

The exploratory analysis showed that recent realised volatility has a strong positive relationship with future realised volatility.

Therefore, simply using recent volatility as a forecast already exploits one of the most important properties of financial volatility: **persistence**.

The baseline results show that this information is useful but incomplete.

The difference between MAE and RMSE suggests that some predictions contain much larger errors than the typical observation.

These large errors are likely associated with sudden transitions between market regimes.

For example:

```text
calm market
    ↓
unexpected shock
    ↓
large volatility increase
```

A persistence forecast reacts only after volatility has already increased.

It may therefore struggle during the beginning of stress episodes.

This creates the main challenge for the subsequent models:

> Can additional information such as VIX, recent equity returns, and interest-rate variables improve the forecast before or during volatility regime changes?

---

# Planned Models

The first modelling version compares three approaches:

```text
1. Persistence baseline

2. Regularised linear regression
   Ridge / Elastic Net

3. Nonlinear gradient boosting model
   LightGBM
```

The model set is intentionally compact.

The objective is not to test a large number of algorithms but to compare increasingly sophisticated forecasting approaches.

---

## Persistence

The persistence model represents the simplest meaningful financial benchmark.

It tests whether recent realised volatility alone is sufficient.

---

## Regularised Linear Model

A Ridge or Elastic Net regression will provide a linear benchmark using the full feature set.

This will test whether combining variables such as:

```text
VIX
recent returns
recent realised volatility
Treasury yields
yield curve
rate changes
```

improves the forecast through linear relationships.

Regularisation is useful because several volatility-related features are likely to be correlated with each other.

For example:

```text
volatility_5d
volatility_10d
volatility_20d
```

contain overlapping information.

---

## LightGBM

LightGBM will be used as the nonlinear model.

Unlike linear regression, it can capture:

- nonlinear relationships;
- feature interactions;
- threshold effects;
- different behaviour across market regimes.

For example, the effect of a VIX increase may depend on whether VIX is already:

```text
low
moderate
or
extremely elevated
```

These relationships may not be captured by a simple linear model.

---

# Walk-Forward Validation

The final models will be evaluated using chronological walk-forward validation.

A typical expanding-window structure is:

```text
Fold 1

Train:
historical data ──────────────►

Test:
                               future period


Fold 2

Train:
historical data ─────────────────────►

Test:
                                      next future period
```

An example using yearly folds could be:

```text
Fold 1
Train: data before 2022
Test:  2022

Fold 2
Train: data before 2023
Test:  2023

Fold 3
Train: data before 2024
Test:  2024

Fold 4
Train: data before 2025
Test:  2025
```

The training window expands as new historical information becomes available.

At every stage:

```text
training dates < test dates
```

The model is therefore always evaluated on observations that occur after the data used to train it.

This is much closer to how the forecasting model would operate in practice.

---

## Fair Model Comparison

Persistence, Ridge, and LightGBM will all be evaluated on the **same test periods**.

This is important because comparing:

```text
Model A on one period
```

with:

```text
Model B on another period
```

would not provide a fair comparison.

Some periods are much more difficult to forecast than others.

For example:

```text
calm market year
```

may naturally produce lower errors than:

```text
crisis / stress year
```

All models will therefore receive exactly the same out-of-sample observations.

The initial persistence metrics reported above are descriptive full-sample results.

The final headline baseline performance will be recomputed using the same walk-forward test folds as the machine-learning models.

---

# Stress-Regime Evaluation

Average forecasting metrics do not necessarily reveal how a model behaves during the periods that matter most for financial risk.

The project will therefore evaluate forecasts separately during high-volatility periods.

A stress threshold can be constructed from a high quantile of the target distribution.

For example:

```text
Normal:
target below 70th percentile

Elevated:
target between 70th and 90th percentile

Stress:
target above 90th percentile
```

The thresholds must be estimated using the **training period only**.

Using the complete dataset would introduce information about the future distribution of the target.

Potential stress-period evaluation metrics include:

- MAE during stress periods;
- RMSE during stress periods;
- stress-regime recall;
- confusion matrix;
- number of major volatility spikes underestimated by the model.

This is especially important because a model could achieve good average MAE while performing poorly during extreme market events.

---

# Model Interpretation

Once the final forecasting models have been trained, feature importance and SHAP values can be used to investigate which variables contribute most to predictions.

Interpretation will focus particularly on:

```text
VIX level
recent realised volatility
recent S&P 500 returns
VIX changes
yield curve
Treasury-yield changes
```

SHAP values will be interpreted as explanations of the model's predictions.

They should not be interpreted as evidence of causal relationships.

---

# Main Modelling Questions

The project focuses on two main questions.

### 1. Can machine learning beat volatility persistence?

The first question is:

> Can a linear or nonlinear machine-learning model improve five-day forward realised-volatility forecasts relative to simply using recent realised volatility?

The persistence baseline provides the reference point.

### 2. Can additional market indicators help during stress periods?

The second question is:

> Does combining recent realised volatility with VIX, equity returns, and interest-rate information improve forecasting performance during market-stress episodes?

This question is particularly important because the exploratory analysis showed that some extreme volatility events were preceded by strong stress signals while others occurred more abruptly.

---

# Current Results

The project currently reaches the persistence-baseline stage.

The initial results are:

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence baseline | 0.1820 | 0.2641 |

These values will later be replaced or complemented by the final **walk-forward out-of-sample results** so that all forecasting models can be compared under the same evaluation framework.