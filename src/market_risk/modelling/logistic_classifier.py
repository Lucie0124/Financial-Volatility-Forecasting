"""Refactor stable logistic classifier code from the notebook: 05_regime_clssification.ipynb"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, ConfusionMatrixDisplay

from market_risk.modelling.dataset import TEST_YEARS, load_model_dataset, prepare_train_test, create_walk_forward_fold
from market_risk.evaluation.regimes import get_regime_threshold, classify_regime

def create_logistic_model():
    logistic_model = Pipeline(
        steps=[("scaler", StandardScaler()),
               ("classifier", LogisticRegression(class_weight="balances", max_iter=1000, random_state=42))      
        ]
    )
    
    
