"""Refactor stable LightGBM classifier code from the notebook: 05_regime_clssification.ipynb"""

from lightgbm import LGBMClassifier


def create_lightgbm_classifier():
    model = LGBMClassifier(
        ojective="multiclass",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    return model

def create_lightgbm_classifier():
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    return model 