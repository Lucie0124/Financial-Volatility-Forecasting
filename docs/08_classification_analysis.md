# Future-regime classification

For each year, q70 and q90 of the purged complete training targets define
Normal, Elevated and Stress. Threshold equality enters the higher regime.
Standardisation and threshold estimation use training data only. Annual
models are recreated independently. Scaling every target by a common positive
constant leaves these quantile-based labels unchanged.

The [evaluation module](../src/market_risk/evaluation/evaluate_classifiers.py)
exports Dummy, Logistic Regression and LightGBM predictions, class-named
probabilities, annual/pooled metrics and confusion counts. The
[notebook](../notebooks/05_regime_classification.ipynb) analyses these exports
without mutable training loops or intermediate metric redefinitions.

## Pooled results

| model | count | accuracy | macro_f1 | stress_precision | stress_recall | stress_f1 | stress_support | predicted_stress_count | stress_roc_auc | stress_average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | 1003 | 0.635095 | 0.258943 | 0.000000 | 0.000000 | 0.000000 | 107 | 0 | 0.500000 | 0.106680 |
| Logistic Regression | 1003 | 0.515454 | 0.478487 | 0.323232 | 0.897196 | 0.475248 | 107 | 297 | 0.882145 | 0.408111 |
| LightGBM | 1003 | 0.554337 | 0.506225 | 0.278481 | 0.822430 | 0.416076 | 107 | 316 | 0.828042 | 0.351546 |

Dummy predicts Normal on all test dates and has the highest accuracy. It
never detects Stress. LightGBM has the best fixed-three-class macro-F1;
Logistic Regression detects more Stress with fewer false Stress alerts in
this historical experiment. These address different objectives, so there is
no universal best classifier.

Logistic Regression detects 96 of 107 Stress dates: 1 becomes Normal and
10 become Elevated. Its 297 Stress alerts include 201 false positives.
LightGBM detects 88 Stress dates: 3 become Normal, 16 become Elevated, and
228 of its 316 Stress alerts are false positives.

## Undefined metrics and ranking scores

The 2023 test year contains no actual Stress: Stress recall, ROC-AUC and
average precision are exported as missing. In 2024, Stress support is five
and recall is genuinely zero for both fitted classifiers. Macro-F1 always
includes all three labels; undefined precision/F1 use zero.

`stress_average_precision` is sklearn average precision, not trapezoidal
PR-AUC. Dummy's constant Stress score has ROC-AUC 0.5 and average precision
equal to pooled Stress prevalence, 0.106680. Probability columns follow
estimator class names rather than assumed positions.

Pooled metrics are recalculated from predictions, not averages of annual
scores. Pooled ranking combines annual models and changing regime thresholds.
Ranking ability does not establish calibrated probabilities or the value
of an alert after decision costs. Class balancing changes the training
objective and requires care when interpreting output probabilities.

## Final artifacts

The final Logistic Regression uses 2484 complete
labelled rows and thresholds q70=0.158424 and
q90=0.247220. These thresholds use the full
training history through 2026; they are not substituted into historical
evaluation folds. Artifacts are fitted candidates and are not a new test.
