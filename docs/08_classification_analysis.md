```markdown
# Future Volatility-Regime Classification Analysis

## Objective

The classification task predicts the future volatility regime over the next five trading days using only information available at time \(t\).

The target classes are:

- **Normal**
- **Elevated**
- **Stress**

Regime thresholds are recalculated within each walk-forward fold using the training target distribution only:

- **Normal:** below the training 70th percentile
- **Elevated:** between the training 70th and 90th percentiles
- **Stress:** at or above the training 90th percentile

The same leakage-safe expanding walk-forward framework used for regression is retained, including the five-day purge at each train/test boundary.

---

## Out-of-Sample Regime Distribution

The pooled 2022–2025 test set contains:

| Regime | Observations |
|---|---:|
| Normal | 637 |
| Elevated | 259 |
| Stress | 107 |

The regime distribution varies substantially across years.

- **2022** is dominated by Elevated and Stress observations, with only 26 Normal observations.
- **2023** is much calmer, with 210 Normal observations and no Stress observations.
- **2024** is also mostly Normal, with only 5 Stress observations.
- **2025** contains 186 Normal, 47 Elevated, and 17 Stress observations.

This confirms that the classification problem is strongly affected by changing market environments.

---

## Dummy Classifier Baseline

A `DummyClassifier(strategy="most_frequent")` is used as the classification baseline.

The dummy model predicts the most frequent training class for every test observation. Since the training distribution is generally dominated by the Normal regime, the baseline mainly predicts Normal.

This demonstrates why accuracy alone is not an appropriate evaluation metric for this task. In calm years, the dummy model can achieve relatively high accuracy while having zero ability to detect Stress.

For example, in 2025:

| Model | Accuracy | Macro F1 | Stress Recall |
|---|---:|---:|---:|
| Dummy | 0.744 | 0.284 | 0.000 |
| Logistic Regression | 0.684 | 0.534 | 0.647 |

Although the DummyClassifier achieves higher accuracy, it fails to identify any Stress observations.

For this reason, the main classification metrics are:

- Macro F1
- Stress precision
- Stress recall
- Stress F1
- confusion matrices
- ROC-AUC and PR-AUC for Stress probability

Accuracy is treated as a secondary metric.

---

## Logistic Regression

The Logistic Regression classifier uses a `StandardScaler` followed by multiclass Logistic Regression with balanced class weights.

The balanced class weights reduce the dominance of the Normal class and place greater importance on the rarer Elevated and Stress observations.

### Walk-Forward Results

| Year | Accuracy | Macro F1 | Stress Precision | Stress Recall | Stress F1 |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.430 | 0.278 | 0.390 | 1.000 | 0.561 |
| 2023 | 0.216 | 0.155 | 0.000 | N/A | 0.000 |
| 2024 | 0.730 | 0.374 | 0.000 | 0.000 | 0.000 |
| 2025 | 0.684 | 0.534 | 0.306 | 0.647 | 0.415 |

Performance varies considerably across years.

In 2022, Logistic Regression is highly sensitive to Stress and identifies all 85 Stress observations, but it also produces many false Stress predictions.

In 2023, there are no actual Stress observations, so Stress recall is not meaningfully defined.

In 2024, the model misses all five Stress observations.

In 2025, the classifier detects 11 of the 17 Stress observations, giving a Stress recall of approximately 64.7%.

### Pooled Logistic Regression Results

Across all 1,003 out-of-sample observations:

| Metric | Value |
|---|---:|
| Accuracy | 0.515 |
| Macro F1 | 0.478 |
| Stress Precision | 0.323 |
| Stress Recall | 0.897 |
| Stress F1 | 0.475 |

The pooled class-level results are:

| Regime | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.90 | 0.52 | 0.66 |
| Elevated | 0.27 | 0.35 | 0.30 |
| Stress | 0.32 | 0.90 | 0.48 |

The strongest result is the high Stress recall.

Out of 107 actual Stress observations:

- 96 are correctly classified as Stress
- 10 are classified as Elevated
- only 1 is classified as Normal

This means the classifier rarely completely misses a future Stress regime.

However, the relatively low Stress precision shows that this sensitivity comes with many false Stress warnings.

The Elevated regime is the most difficult class to separate cleanly.

---

## LightGBM Classifier

A multiclass LightGBM classifier is evaluated using the same features, walk-forward folds, regime thresholds, class weighting, and evaluation metrics.

### Walk-Forward Results

| Year | Accuracy | Macro F1 | Stress Precision | Stress Recall | Stress F1 |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.498 | 0.350 | 0.438 | 0.918 | 0.593 |
| 2023 | 0.308 | 0.258 | 0.000 | N/A | 0.000 |
| 2024 | 0.746 | 0.390 | 0.000 | 0.000 | 0.000 |
| 2025 | 0.664 | 0.566 | 0.400 | 0.588 | 0.476 |

LightGBM generally produces better overall multiclass performance than Logistic Regression, particularly in Macro F1.

However, it is slightly less sensitive to Stress.

### Pooled LightGBM Results

| Metric | Value |
|---|---:|
| Accuracy | 0.554 |
| Macro F1 | 0.506 |
| Stress Precision | 0.278 |
| Stress Recall | 0.822 |
| Stress F1 | 0.416 |

The pooled LightGBM confusion matrix shows that out of 107 Stress observations:

- 88 are correctly classified as Stress
- 16 are classified as Elevated
- 3 are classified as Normal

LightGBM therefore still detects most Stress observations, but it produces more severe Stress misses than Logistic Regression.

---

## Logistic Regression vs LightGBM

The pooled comparison is:

| Model | Accuracy | Macro F1 | Stress Precision | Stress Recall | Stress F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.515 | 0.478 | **0.323** | **0.897** | **0.475** |
| LightGBM | **0.554** | **0.506** | 0.278 | 0.822 | 0.416 |

The two models have different strengths.

### LightGBM

LightGBM provides stronger overall three-class classification performance:

- higher accuracy
- higher Macro F1
- better average separation across Normal, Elevated, and Stress

### Logistic Regression

Logistic Regression provides stronger Stress-focused performance:

- higher Stress precision
- higher Stress recall
- higher Stress F1
- fewer Stress-to-Normal errors

The Stress error analysis confirms this:

| Model | Stress → Normal | Stress → Elevated | False Stress Predictions |
|---|---:|---:|---:|
| Logistic Regression | **1** | **10** | **201** |
| LightGBM | 3 | 16 | 228 |

Logistic Regression therefore not only detects more Stress observations, but also produces fewer severe misses and fewer false Stress predictions in the pooled sample.

---

## Stress Probability Analysis

Both classifiers produce class probabilities.

The mean predicted Stress probability increases consistently with the actual regime:

| Model | Actual Normal | Actual Elevated | Actual Stress |
|---|---:|---:|---:|
| LightGBM | 0.214 | 0.384 | 0.614 |
| Logistic Regression | 0.221 | 0.534 | 0.769 |

This indicates that `prob_stress` contains useful information beyond the final hard class prediction.

Logistic Regression shows stronger separation between actual Normal and actual Stress observations.

---

## Stress ROC-AUC and PR-AUC

Stress is also evaluated as a binary event:

- **Stress = 1**
- **Normal / Elevated = 0**

Using predicted Stress probabilities:

| Model | Stress ROC-AUC | Stress PR-AUC |
|---|---:|---:|
| Logistic Regression | **0.882** | **0.408** |
| LightGBM | 0.828 | 0.352 |

Stress prevalence in the pooled test set is approximately 10.7%, so the no-skill baseline for average precision is approximately 0.107.

Both models therefore provide meaningful Stress discrimination, but Logistic Regression performs better.

The ROC and Precision-Recall curves confirm that Logistic Regression provides the stronger overall ranking of future Stress risk across classification thresholds.

---

## Final Classification Conclusion

The classification results reveal a clear trade-off between general multiclass performance and Stress-specific detection.

**LightGBM is the stronger general-purpose three-class classifier**, with the highest pooled accuracy and Macro F1.

However, **Logistic Regression is the stronger Stress-risk model**. It achieves:

- approximately 89.7% pooled Stress recall
- higher Stress precision and F1 than LightGBM
- only one Stress-to-Normal error across 107 Stress observations
- fewer false Stress predictions
- higher Stress ROC-AUC
- higher Stress PR-AUC

For the intended market-risk application, where failing to identify future Stress is particularly important, Logistic Regression currently provides the more attractive model for Stress-risk prediction.

The modelling tasks therefore have complementary preferred models:

- **Ridge Regression** remains the strongest general-purpose continuous volatility forecaster.
- **Logistic Regression** is the strongest model for future Stress-risk detection.
- **LightGBM** remains useful as a complementary nonlinear model, particularly for broader three-regime classification.

The final application can therefore combine:

- expected future volatility from Ridge Regression
- predicted regime and Stress probability from Logistic Regression
```