# Correction record

The audited defects below have executable fixes. Historical outputs and the
initial audit report remain in Git history or the user's ignored report files.
The current experiment is described by `results/evaluation_manifest.json`.

| Audited defect | Correction and verification |
| --- | --- |
| Forward variance omitted division by five | Shared annualised RMS definition; known-return and horizon-alignment tests |
| Persistence used a different estimator | Same RMS on the five observed returns; no-future-dependence tests |
| Baseline train counts differed | Shared complete-case folds and matching dates/counts across all models |
| Classification depended on notebook state | Executable module and common runner; fresh estimators and training-only scaling/threshold tests |
| Dummy predictions were omitted | Dummy predictions, probabilities, annual/pooled scores and confusion counts exported |
| Intermediate metrics used wrong labels or dates | One-dimensional validated labels and explicit Stress metrics; pooled recomputation |
| Absent Stress was reported as failed recall | Undefined recall/ROC-AUC/average precision exported as missing with support counts |
| Average precision labelled PR-AUC | Explicit `stress_average_precision` name and definition |
| Probability positions assumed class order | Columns mapped using estimator class names; reversed-order and missing-class tests |
| Regime joins could mix stale targets or silently omit dates | Strict one-to-one outer joins, target comparison, recomputed errors and failure tests |
| Old dataset, forecasts and thresholds remained after source changes | Rebuilt modelling target, all 16 result CSVs and final models with source/data/output fingerprints |
| Notebook filenames, execution order and links were broken | Five notebooks read verified exports; fresh-kernel execution stops on any cell error |
| Cleaning relied on earlier sorting | Sort before forward fill; reject duplicate/missing dates and test gap limits |
| Inputs could contain infinite features or invalid prices | Reject infinite model inputs and non-positive/non-finite index levels before fitting |
| Index columns exported as `Unnamed: 0` | All regenerated evaluation tables omit index columns; CSV round-trip verification |
| Final artifacts lacked definition/version provenance | Metadata includes units, horizon, data dates, quantiles, classes, versions and hashes |
| Interrupted overwrites could retain an old completion marker | Invalidate marker before writes; create manifest/metadata last |
| Dependencies lacked bounds, direct joblib and test tools | Bounded runtime dependencies, dev extra and exact Python 3.12 dependency constraints |
| No automated clean execution | GitHub Actions creates an isolated venv, installs the constrained environment, checks tests/fingerprints and executes fresh notebook copies; preinstalled runner packages are excluded |
| Documentation used stale units, gains and conclusions | Rewritten results from current exports; corrected 4.77% MAE gain and explicit limitations |

The following are research extensions, not fixed software defects: nested
temporal tuning, an untouched future holdout, dependence-adjusted uncertainty,
ablations, GARCH/HAR alternatives, probability calibration, episode-based
evaluation, intraday data and a decision backtest with costs. Exact same-day
publication availability remains unverified. The correction does not claim
these studies or an API/dashboard have been completed.
