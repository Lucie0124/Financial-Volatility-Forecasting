"""Reproducible annual walk-forward regime classification, outside notebooks."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from market_risk.evaluation.classification import (
    CLASS_LABELS,
    compute_classification_metrics,
    compute_stress_probability_metrics,
)
from market_risk.evaluation.regimes import classify_regime, get_regime_threshold
from market_risk.modelling.dataset import iter_walk_forward_folds, load_model_dataset
from market_risk.modelling.lightgbm_classifier import create_lightgbm_classifier
from market_risk.modelling.logistic_classifier import create_logistic_model

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "results" / "classification"


def _confusion_rows(actual, predicted, model, scope, year=None):
    matrix = confusion_matrix(actual, predicted, labels=CLASS_LABELS)
    return [
        {
            "scope": scope,
            "test_year": year,
            "model": model,
            "actual_regime": actual_label,
            "predicted_regime": predicted_label,
            "count": int(matrix[i, j]),
        }
        for i, actual_label in enumerate(CLASS_LABELS)
        for j, predicted_label in enumerate(CLASS_LABELS)
    ]


def evaluate_classifiers_walk_forward(df, model_factories=None):
    """Return annual metrics, predictions, pooled metrics and confusion counts.

    Each factory creates a fresh estimator for every fold. Thresholds and
    scalers use training data only. Dummy predictions and probabilities are
    retained alongside the fitted classifiers, including missing-class folds.
    """
    if model_factories is None:
        model_factories = {
            "Dummy": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": create_logistic_model,
            "LightGBM": create_lightgbm_classifier,
        }
    if not model_factories:
        raise ValueError("At least one classifier factory is required.")

    metric_rows, prediction_frames, confusion_rows = [], [], []
    for fold in iter_walk_forward_folds(df):
        elevated, stress = get_regime_threshold(fold.y_train)
        train_labels = fold.y_train.map(
            lambda value: classify_regime(value, elevated, stress)
        )
        test_labels = fold.y_test.map(
            lambda value: classify_regime(value, elevated, stress)
        )

        for name, factory in model_factories.items():
            model = factory()
            model.fit(fold.X_train, train_labels)
            predicted = model.predict(fold.X_test)
            # Estimator classes are not in CLASS_LABELS order. Map by name,
            # assigning zero probability to a class absent from training.
            probabilities = pd.DataFrame(
                model.predict_proba(fold.X_test),
                columns=model.classes_,
                index=fold.test.index,
            ).reindex(columns=CLASS_LABELS, fill_value=0.0)
            metric_rows.append({
                "test_year": fold.test_year,
                "model": name,
                "train_size": len(fold.train),
                "test_size": len(fold.test),
                "train_start": fold.train["date"].min(),
                "train_end": fold.train["date"].max(),
                "test_start": fold.test["date"].min(),
                "test_end": fold.test["date"].max(),
                "elevated_threshold": elevated,
                "stress_threshold": stress,
                **compute_classification_metrics(test_labels, predicted),
                **compute_stress_probability_metrics(
                    test_labels, probabilities["Stress"]
                ),
            })
            prediction_frames.append(pd.DataFrame({
                "date": fold.test["date"].to_numpy(),
                "test_year": fold.test_year,
                "model": name,
                "y_true": fold.y_test.to_numpy(),
                "actual_regime": test_labels.to_numpy(),
                "predicted_regime": predicted,
                "prob_normal": probabilities["Normal"].to_numpy(),
                "prob_elevated": probabilities["Elevated"].to_numpy(),
                "prob_stress": probabilities["Stress"].to_numpy(),
                "elevated_threshold": elevated,
                "stress_threshold": stress,
            }))
            confusion_rows.extend(_confusion_rows(
                test_labels, predicted, name, "fold", fold.test_year
            ))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled_rows = []
    for name, group in predictions.groupby("model", sort=False):
        pooled_rows.append({
            "model": name,
            "count": len(group),
            **compute_classification_metrics(
                group["actual_regime"], group["predicted_regime"]
            ),
            **compute_stress_probability_metrics(
                group["actual_regime"], group["prob_stress"]
            ),
        })
        confusion_rows.extend(_confusion_rows(
            group["actual_regime"], group["predicted_regime"], name, "pooled"
        ))

    matrices = pd.DataFrame(confusion_rows)
    matrices["test_year"] = matrices["test_year"].astype("Int64")
    return (
        pd.DataFrame(metric_rows),
        predictions,
        pd.DataFrame(pooled_rows),
        matrices,
    )


def main():
    annual, predictions, pooled, matrices = evaluate_classifiers_walk_forward(
        load_model_dataset()
    )
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for filename, frame in {
        "fold_results.csv": annual,
        "predictions.csv": predictions,
        "pooled_metrics.csv": pooled,
        "confusion_matrices.csv": matrices,
    }.items():
        frame.to_csv(OUTPUT_PATH / filename, index=False)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
