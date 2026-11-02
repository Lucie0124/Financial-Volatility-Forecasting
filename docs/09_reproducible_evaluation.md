# Reproducing the evaluation

Run from the repository root after installation:

```bash
python -m market_risk.features.build_target
python -m market_risk.evaluation.run_evaluations
```

The evaluation command checks the full target before splitting. It uses common
complete-case rows for persistence, Ridge, LightGBM, Dummy classification,
Logistic Regression and LightGBM classification. Five forecast origins are
purged before each test year. Thresholds and fitted scalers use training rows
only. Models retain their fixed settings; there is no tuning on test years.

For a separate experiment, use `--dataset path/to/model_dataset.csv` and
`--output-dir path/to/results`. All computations finish before exports begin.
The command writes 16 CSVs and an `evaluation_manifest.json` with the input
dataframe hash, source hashes, output hashes, package versions and definitions.
The manifest is written last. Verify its output hashes before combining files
from different runs. Independent model entry points do not refresh this manifest.

Classification exports contain Dummy predictions, class probabilities mapped
by class name, annual and pooled metrics, and confusion counts. Pooled metrics
are recomputed from all predictions. Macro-F1 always includes the three labels.
Stress recall is missing when there are no actual Stress observations; ROC-AUC
is missing if only one binary outcome is present. Average precision is named
explicitly and is missing without positives. Undefined precision/F1 use zero.

Regime comparison requires exactly matching dates and true targets; missing,
extra or duplicate predictions and legacy target scales fail rather than being
silently joined. Signed errors are always recomputed as `y_true - y_pred`.

The final models in `artifacts/` are fitted separately on all complete labelled
rows. They are not the annual models used to obtain the walk-forward scores.
The years 2022–2025 have already been inspected and support descriptive
historical comparison, not a new untouched holdout or a claim of significance.
