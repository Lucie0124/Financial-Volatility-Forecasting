"""Evaluate every fixed model in one run and export inspectable results."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from market_risk.evaluation.evaluate_classifiers import evaluate_classifiers_walk_forward
from market_risk.evaluation.regimes import (
    build_test_regimes,
    combine_predictions_with_regimes,
    evaluate_by_regime,
)
from market_risk.evaluation.walk_forward import walk_forward_evaluation
from market_risk.features.volatility import TRADING_DAYS_PER_YEAR, VOLATILITY_WINDOW
from market_risk.modelling.dataset import FEATURES, load_model_dataset
from market_risk.modelling.lightgbm_model import evaluate_lightgbm_walk_forward
from market_risk.modelling.ridge import evaluate_ridge_walk_forward

ROOT_PATH = Path(__file__).resolve().parents[3]


def evaluate_all_models(df):
    """Compute all outputs before writing any file; retain fixed model settings."""
    persistence_folds, persistence = walk_forward_evaluation(df)
    ridge_folds, ridge, coefficients = evaluate_ridge_walk_forward(df)
    lightgbm_folds, lightgbm, importances = evaluate_lightgbm_walk_forward(df)
    predictions = {
        "persistence": persistence, "ridge": ridge, "lightgbm": lightgbm
    }
    regimes = build_test_regimes(df)
    combined = combine_predictions_with_regimes(regimes, predictions)
    classification_folds, classification_predictions, pooled, matrices = (
        evaluate_classifiers_walk_forward(df)
    )
    regression_metrics = pd.DataFrame([
        {
            "model": model,
            "count": len(frame),
            "MAE": mean_absolute_error(frame["y_true"], frame["y_pred"]),
            "RMSE": root_mean_squared_error(frame["y_true"], frame["y_pred"]),
        }
        for model, frame in predictions.items()
    ])
    return {
        "persistence/fold_results.csv": persistence_folds,
        "persistence/predictions.csv": persistence,
        "ridge/fold_results.csv": ridge_folds,
        "ridge/predictions.csv": ridge,
        "ridge/coefficients.csv": coefficients,
        "lightgbm/results.csv": lightgbm_folds,
        "lightgbm/predictions.csv": lightgbm,
        "lightgbm/feature_importances.csv": importances,
        "regression_metrics.csv": regression_metrics,
        "regime_evaluation/regimes.csv": regimes,
        "regime_evaluation/predictions_by_regime.csv": combined,
        "regime_evaluation/regime_metrics.csv": evaluate_by_regime(combined),
        "classification/fold_results.csv": classification_folds,
        "classification/predictions.csv": classification_predictions,
        "classification/pooled_metrics.csv": pooled,
        "classification/confusion_matrices.csv": matrices,
    }


def save_evaluation_results(tables, df, output_dir):
    """Export CSVs without index columns and write a completion manifest last."""
    output_dir = Path(output_dir)
    # An interrupted overwrite must not leave a previous completion marker.
    (output_dir / 'evaluation_manifest.json').unlink(missing_ok=True)
    hashes = {}
    for relative_path, frame in tables.items():
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator='\n')
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Record the exact source files as well as the input dataframe; a Git HEAD
    # alone would not describe an evaluation performed with uncommitted edits.
    code_hashes = {
        path.relative_to(ROOT_PATH).as_posix(): hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
        for path in sorted((ROOT_PATH / "src" / "market_risk").rglob("*.py"))
    }
    dataset_bytes = df.to_csv(index=False, lineterminator='\n').encode("utf-8")
    manifest = {
        "status": "complete",
        "target_definition": "sqrt((252 / 5) * sum(r[t+1:t+6]**2))",
        "persistence_definition": "sqrt((252 / 5) * sum(r[t-4:t+1]**2))",
        "horizon_sessions": VOLATILITY_WINDOW,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "features": FEATURES,
        "test_years": tables["persistence/fold_results.csv"]["test_year"].tolist(),
        "data_rows": len(df),
        "data_start": df["date"].min().strftime("%Y-%m-%d"),
        "data_end": df["date"].max().strftime("%Y-%m-%d"),
        "dataframe_csv_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in ["numpy", "pandas", "scikit-learn", "lightgbm"]
        },
        "source_sha256": code_hashes,
        "output_sha256": hashes,
        "metric_conventions": {
            "macro_f1": "fixed Normal/Elevated/Stress labels; undefined F1 is zero",
            "stress_recall": "missing when no actual Stress is present",
            "stress_roc_auc": "missing when only one binary outcome is present",
            "stress_average_precision": "sklearn average precision; missing without positives",
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline='\n'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Corrected modelling CSV; defaults to the repository snapshot.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_PATH / "results")
    args = parser.parse_args(argv)
    df = load_model_dataset(args.dataset)
    tables = evaluate_all_models(df)
    save_evaluation_results(tables, df, args.output_dir)
    print(tables["regression_metrics.csv"].to_string(index=False))
    print(tables["classification/pooled_metrics.csv"].to_string(index=False))
    print(f"Results and manifest saved to {args.output_dir}")


if __name__ == "__main__":
    main()
