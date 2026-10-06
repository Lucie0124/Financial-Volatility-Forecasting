"""Refactor stable logistic classifier code from the notebook: 05_regime_classification.ipynb"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_model():
    logistic_model = Pipeline(
        steps=[("scaler", StandardScaler()),
               ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))      
        ]
    )
    return logistic_model
    
    
