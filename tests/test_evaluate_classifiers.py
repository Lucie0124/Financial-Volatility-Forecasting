import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from market_risk.evaluation.evaluate_classifiers import evaluate_classifiers_walk_forward
from market_risk.features.build_target import build_target
from market_risk.modelling.dataset import iter_walk_forward_folds
from market_risk.modelling.logistic_classifier import create_logistic_model


def test_all_models_including_dummy_are_recorded_with_training_only_scalers(corrected_dataset):
    models = []

    def logistic_factory():
        model = create_logistic_model()
        models.append(model)
        return model

    annual, predictions, pooled, matrices = evaluate_classifiers_walk_forward(
        corrected_dataset,
        {
            "Dummy": lambda: DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": logistic_factory,
        },
    )

    assert len(annual) == 4
    assert set(annual["model"]) == {"Dummy", "Logistic Regression"}
    assert set(pooled["model"]) == set(annual["model"])
    assert not predictions.duplicated(["model", "date"]).any()
    np.testing.assert_allclose(
        predictions[["prob_normal", "prob_elevated", "prob_stress"]].sum(axis=1), 1.0
    )
    folds = list(iter_walk_forward_folds(corrected_dataset))
    assert len(models) == len(folds)
    for model, fold in zip(models, folds):
        np.testing.assert_allclose(
            model.named_steps["scaler"].mean_, fold.X_train.mean().to_numpy()
        )
        rows = annual.loc[annual["test_year"] == fold.test_year]
        np.testing.assert_allclose(rows["elevated_threshold"], fold.y_train.quantile(0.7))
        np.testing.assert_allclose(rows["stress_threshold"], fold.y_train.quantile(0.9))
    for name, group in predictions.groupby("model"):
        result = pooled.set_index("model").loc[name]
        true_stress = group["actual_regime"].eq("Stress")
        caught = true_stress & group["predicted_regime"].eq("Stress")
        assert result["count"] == len(group)
        assert result["accuracy"] == pytest.approx(
            group["actual_regime"].eq(group["predicted_regime"]).mean()
        )
        assert result["stress_recall"] == pytest.approx(caught.sum() / true_stress.sum())
        counts = matrices.loc[(matrices["scope"] == "pooled") & (matrices["model"] == name)]
        assert len(counts) == 9
        assert counts["count"].sum() == len(group)


def test_probability_columns_follow_class_names_not_positions(corrected_dataset):
    instances = []

    class ReorderedClassifier:
        classes_ = np.array(["Stress", "Normal", "Elevated"])

        def fit(self, X, y):
            assert y.ndim == 1
            instances.append(self)
            return self

        def predict(self, X):
            return np.repeat("Normal", len(X))

        def predict_proba(self, X):
            return np.tile([0.1, 0.7, 0.2], (len(X), 1))

    _, predictions, _, _ = evaluate_classifiers_walk_forward(
        corrected_dataset, {"Reordered": ReorderedClassifier}
    )

    assert len(instances) == 2
    assert instances[0] is not instances[1]
    np.testing.assert_allclose(predictions["prob_normal"], 0.7)
    np.testing.assert_allclose(predictions["prob_elevated"], 0.2)
    np.testing.assert_allclose(predictions["prob_stress"], 0.1)


def test_test_period_returns_cannot_change_that_folds_thresholds(corrected_dataset):
    factories = {"Dummy": lambda: DummyClassifier(strategy="most_frequent")}
    before, _, _, _ = evaluate_classifiers_walk_forward(corrected_dataset, factories)
    changed = corrected_dataset.copy()
    changed.loc[changed["date"] >= "2022-01-01", "return_1d"] *= 10
    changed = build_target(changed)

    after, _, _, _ = evaluate_classifiers_walk_forward(changed, factories)

    # The 2023 model may learn from 2022, but the 2022 thresholds may not.
    columns = ["elevated_threshold", "stress_threshold"]
    pd.testing.assert_frame_equal(
        before.loc[before["test_year"] == 2022, columns].reset_index(drop=True),
        after.loc[after["test_year"] == 2022, columns].reset_index(drop=True),
    )


def test_dummy_supports_classes_missing_from_training(corrected_dataset):
    df = corrected_dataset.copy()
    df["return_1d"] = 0.01
    df = build_target(df)

    annual, predictions, _, _ = evaluate_classifiers_walk_forward(
        df, {"Dummy": lambda: DummyClassifier(strategy="most_frequent")}
    )

    # With constant targets both quantiles coincide, so every label is Stress.
    assert predictions["actual_regime"].eq("Stress").all()
    np.testing.assert_allclose(predictions[["prob_normal", "prob_elevated"]], 0.0)
    np.testing.assert_allclose(predictions["prob_stress"], 1.0)
    assert annual["stress_roc_auc"].isna().all()
