import hashlib
import json

import joblib
import numpy as np
import pytest

from market_risk.modelling.dataset import FEATURES, TARGET_COLUMN
from market_risk.training.train_final_models import train_and_save


def test_final_models_match_corrected_training_data_and_metadata(corrected_dataset, tmp_path):
    train_and_save(corrected_dataset, tmp_path)
    metadata = json.loads((tmp_path / 'model_metadata.json').read_text())
    complete = corrected_dataset.dropna(subset=FEATURES + [TARGET_COLUMN])
    assert metadata['training_rows'] == len(complete)
    assert metadata['horizon_sessions'] == 5
    assert metadata['regime_thresholds']['stress'] == pytest.approx(complete[TARGET_COLUMN].quantile(.9))
    for filename, checksum in metadata['artifact_sha256'].items():
        path = tmp_path / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        model = joblib.load(path)
        np.testing.assert_allclose(model.named_steps['scaler'].mean_, complete[FEATURES].mean())
        assert model.named_steps['scaler'].n_samples_seen_ == len(complete)
        assert len(model.predict(complete[FEATURES].head(2))) == 2


def test_final_training_rejects_legacy_labels_before_export(corrected_dataset, tmp_path):
    corrected_dataset[TARGET_COLUMN] *= np.sqrt(5)
    output = tmp_path / 'artifacts'
    with pytest.raises(ValueError, match='Regenerate'):
        train_and_save(corrected_dataset, output)
    assert not output.exists()
