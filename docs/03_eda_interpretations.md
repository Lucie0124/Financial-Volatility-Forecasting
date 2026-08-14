# Exploratory Data Analysis

## Overview

The exploratory analysis focuses on the behaviour of the forecasting target and its relationship with the main market indicators.

The objective is not to make causal claims at this stage, but to:

* understand the distribution and temporal behaviour of volatility;
* identify potentially informative predictors;
* inspect extreme market-stress periods;
* detect possible modelling challenges such as skewness, heteroskedasticity, and regime dependence.

---

## S&P 500 Index Over Time

The S&P 500 exhibits a clear long-term upward trend over the sample period, interrupted by several sharp drawdowns and recovery phases.

The most visible stress periods correspond to abrupt market declines, particularly around 2020 and during later episodes of market turbulence.

This confirms that the raw S&P 500 index level is strongly non-stationary.

For this reason, the modelling pipeline relies primarily on returns, realised volatility, and changes in market indicators rather than on the raw index level itself.

The plot also suggests that market instability varies substantially through time, with rapid drawdowns generally associated with larger day-to-day price fluctuations.

---

## Five-Day Forward Realised Volatility

The five-day forward realised-volatility series is highly uneven through time.

Most observations remain at relatively low or moderate levels, while a small number of periods exhibit very large volatility spikes.

The most extreme episode occurs around March 2020, when annualised forward realised volatility temporarily rises above 300%.

Other clusters of elevated volatility are also visible around 2018–2019, 2022, and 2025.

Two important features emerge from the time series.

### Volatility Clustering

High-volatility observations tend to occur close to other high-volatility observations rather than appearing uniformly through time.

This suggests strong persistence in volatility dynamics.

In practical terms:

```text
calm periods tend to be followed by calm periods
stress periods tend to be followed by other high-volatility periods
```

This motivates the inclusion of recent realised-volatility features such as:

```text
volatility_5d
volatility_10d
volatility_20d
```

and also justifies the use of a persistence forecast as the first benchmark model.

### Extreme Market Stress

A small number of observations are much larger than typical volatility levels.

These observations should not automatically be removed as outliers because they correspond to genuine market-stress episodes and are directly relevant to the project objective.

The modelling framework should therefore evaluate both average forecasting accuracy and performance during stress periods.

---

## Distribution of Forward Realised Volatility

The distribution of `target_volatility_5d` is strongly right-skewed.

Most observations are concentrated at relatively low or moderate annualised volatility levels, while a small number of stress periods generate very large values.

The distribution contains a long right tail, with a few observations exceeding 100% annualised volatility and extreme episodes above 300%.

This has several implications for model evaluation:

* **MAE** is useful because it is less dominated by a small number of extreme observations.
* **RMSE** should also be reported because it penalises large forecasting errors more strongly.
* Forecasting performance should be evaluated separately during high-volatility regimes.
* Extreme stress observations should remain in the main dataset because they represent economically meaningful events rather than data errors.

The shape of the target distribution also suggests that model errors are unlikely to be uniform across market regimes.

---

## VIX vs Forward Realised Volatility Over Time

The VIX and subsequent five-day realised volatility show a clear visual relationship.

Periods of elevated VIX generally occur close to periods of elevated realised volatility, particularly during major market-stress episodes.

However, realised volatility is considerably more variable and can exhibit sharper short-lived spikes.

The two quantities represent different concepts:

```text
VIX
-> volatility implied by S&P 500 option prices

Forward realised volatility
-> volatility actually observed over the following five trading days
```

The visual co-movement suggests that the VIX may contain useful predictive information about future realised volatility.

However, the time-series plot alone does not establish forecasting ability. Predictive value must be tested using out-of-sample walk-forward evaluation.

---

## Correlation with the Forecasting Target

Pearson correlations were computed between each numerical feature and `target_volatility_5d`.

The strongest positive relationships are observed for:

```text
vix
volatility_5d
volatility_10d
volatility_20d
```

These variables all display substantial positive correlation with future realised volatility.

### VIX

The VIX has the strongest positive linear relationship with the target.

This is consistent with the interpretation of the VIX as a market-implied measure of expected S&P 500 volatility.

Higher VIX levels tend to be associated with higher volatility subsequently realised over the next five trading days.

### Recent Realised Volatility

The 5-, 10-, and 20-day realised-volatility features also show strong positive correlations with the target.

This provides empirical evidence of volatility persistence in the dataset:

> when the market has recently been volatile, the following week is also more likely to be volatile.

The similar correlation levels across several volatility windows indicate that these features contain overlapping information, while still representing different short- and medium-term horizons.

### VIX Changes

The five-day VIX change has a moderate positive relationship with the target, while the one-day VIX change is weaker.

This suggests that the absolute level of market-implied volatility may be more informative than a single short-term change.

However, VIX changes are retained because they may still contribute through nonlinear effects or interactions.

### Equity Returns

Recent S&P 500 returns are negatively correlated with future volatility.

The five-day return shows a stronger negative relationship than the one-day return.

This indicates that recent market declines tend to be followed by higher future realised volatility.

This is consistent with the asymmetric relationship often observed between equity-market losses and volatility.

### Treasury and Yield-Curve Features

Treasury yields, yield changes, and the yield-curve slope show much weaker linear correlations with the short-horizon volatility target.

This suggests that they contain less direct linear information about one-week equity volatility than VIX or recent realised volatility.

They are nevertheless retained for modelling because correlation measures only linear association and does not capture nonlinear relationships or interactions.

### Limitation of Correlation Analysis

Correlation should be interpreted only as an exploratory diagnostic.

A low correlation does not imply that a feature is useless.

Nonlinear models such as LightGBM may extract predictive information from variables that display weak unconditional linear correlation.

Correlation should also not be interpreted as causation.

---

## Scatter Plot: VIX vs Forward Realised Volatility

A scatter plot was used to examine the relationship between VIX and `target_volatility_5d` in more detail.

Each point represents one trading day:

```text
x-axis -> VIX observed at day t
y-axis -> realised volatility over days t+1 to t+5
```

The scatter plot complements the correlation coefficient by revealing the shape and dispersion of the relationship.

### Positive Association

The plot shows a clear upward relationship.

At low VIX levels, future realised volatility is generally low.

As VIX increases, higher realised-volatility outcomes become substantially more common.

This is consistent with the strong positive correlation observed in the correlation analysis.

### Imperfect Relationship

The points are not concentrated around a single line.

For a given VIX level, the subsequently realised volatility can still vary considerably.

This means that VIX alone does not fully determine future volatility.

This provides motivation for combining VIX with additional predictors such as:

```text
recent realised volatility
recent equity returns
interest-rate changes
yield-curve information
```

### Heteroskedasticity

The spread of the observations appears to increase as VIX rises.

At low VIX levels, future volatility values are relatively concentrated.

At high VIX levels, the range of possible future realised-volatility outcomes becomes much wider.

This suggests heteroskedasticity:

> the uncertainty around future volatility itself increases during already stressed market conditions.

This is an important modelling challenge because prediction errors may become larger precisely during the periods that matter most for risk management.

### Extreme Observations

The largest future-volatility observations occur mostly at elevated VIX levels.

However, some large realised-volatility outcomes also occur when VIX is only moderately elevated.

These observations are likely to be among the most difficult cases for the forecasting models.

They should later be examined during forecast-error analysis.

---

## Inspection of Extreme Volatility Periods

The 20 largest values of `target_volatility_5d` were inspected together with:

```text
vix
volatility_20d
yield_curve
```

The observations are strongly concentrated in a small number of market-stress episodes.

### March 2020

Most of the largest target values occur during March 2020.

During these observations:

* VIX is already extremely elevated;
* recent 20-day realised volatility is rising rapidly;
* very high forward realised volatility persists over multiple consecutive dates.

This supports the idea of volatility clustering and indicates that the most extreme 2020 stress regime was accompanied by clearly observable stress signals.

The repeated dates should not be interpreted as independent crises.

Because the forecasting target uses overlapping five-day windows, consecutive rows during the same crisis can all produce very large target values.

### April 2025

A second cluster appears in April 2025.

This episode differs from March 2020.

For example, one of the largest subsequent realised-volatility observations occurs when:

```text
VIX ≈ 21.5
20-day realised volatility ≈ 0.20
```

These indicators are much lower than those observed during the March 2020 crisis, yet the following five-day realised volatility becomes extremely high.

This suggests that not all stress episodes develop with the same observable warning signals.

Some extreme volatility events may occur after clearly elevated VIX and realised volatility, while others may arrive with weaker preceding signals.

These observations are especially relevant for future model-error analysis.

### Yield Curve During Stress Episodes

The yield curve remains positive across the majority of the most extreme observations and varies relatively little compared with VIX and realised volatility.

This is consistent with its weaker unconditional correlation with the short-horizon forecasting target.

The yield curve may still provide regime-specific information, but it does not appear to be a primary direct predictor of the largest volatility spikes.

---

## Main EDA Conclusions

The exploratory analysis provides several consistent findings:

```text
Target distribution
-> strongly right-skewed
-> rare but extreme stress observations

Time-series behaviour
-> strong volatility clustering
-> market regimes are persistent

VIX
-> strongest linear relationship with future volatility
-> useful but imperfect predictor

Recent realised volatility
-> strong positive relationship with the target
-> supports a persistence baseline

Recent S&P 500 returns
-> negative relationship with future volatility
-> recent market declines are associated with higher subsequent volatility

Treasury variables
-> weaker direct linear relationships
-> may still add nonlinear or regime-dependent information

Stress episodes
-> heterogeneous
-> some are preceded by obvious stress signals
-> others are more difficult to anticipate
```

Overall, the EDA suggests that future realised volatility is forecastable to some extent from recent volatility and market-implied uncertainty, but the relationship is nonlinear, noisy, and regime-dependent.

This provides a strong motivation for comparing:

1. a persistence baseline;
2. a regularised linear model;
3. a nonlinear LightGBM model.

The next stage of the project therefore focuses on establishing the persistence baseline before introducing more complex models.
