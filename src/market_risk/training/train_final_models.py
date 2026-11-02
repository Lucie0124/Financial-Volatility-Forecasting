"""Fit deployment candidates on all labelled data, separately from evaluation."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib

from market_risk.evaluation.regimes import classify_regime, get_regime_threshold
from market_risk.modelling.dataset import (
    FEATURES, TARGET_COLUMN, load_model_dataset, validate_modelling_data,
)
from market_risk.modelling.logistic_classifier import create_logistic_model
from market_risk.modelling.ridge import build_ridge_model

ROOT_PATH = Path(__file__).resolve().parents[3]
ARTIFACTS_PATH = ROOT_PATH / 'artifacts'


def prepare_final_training_data(df):
    validate_modelling_data(df)
    training_df = df.dropna(subset=FEATURES + [TARGET_COLUMN]).copy()
    if training_df.empty:
        raise ValueError('No complete labelled rows for final training.')
    return training_df, training_df[FEATURES], training_df[TARGET_COLUMN]


def train_final_ridge(X, y):
    model = build_ridge_model()
    model.fit(X, y)
    return model


def train_final_classifier(X, y):
    elevated, stress = get_regime_threshold(y)
    labels = y.map(lambda value: classify_regime(value, elevated, stress))
    model = create_logistic_model()
    model.fit(X, labels)
    return model, elevated, stress


def save_artifacts(ridge_model, classifier_model, metadata, output_dir=ARTIFACTS_PATH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / 'model_metadata.json'
    metadata_path.unlink(missing_ok=True)
    hashes = {}
    for filename, model in {
        'ridge_model.joblib': ridge_model,
        'logistic_classifier.joblib': classifier_model,
    }.items():
        path = output_dir / filename
        joblib.dump(model, path)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {**metadata, 'artifact_sha256': hashes}
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def train_and_save(df, output_dir=ARTIFACTS_PATH):
    training_df, X, y = prepare_final_training_data(df)
    ridge = train_final_ridge(X, y)
    classifier, elevated, stress = train_final_classifier(X, y)
    metadata = {
        'schema_version': 2,
        'features': FEATURES,
        'target': TARGET_COLUMN,
        'target_definition': 'sqrt((252 / 5) * sum(r[t+1:t+6]**2))',
        'target_unit': 'annualised decimal volatility',
        'horizon_sessions': 5,
        'training_rows': len(training_df),
        'training_start': training_df['date'].min().strftime('%Y-%m-%d'),
        'training_end': training_df['date'].max().strftime('%Y-%m-%d'),
        'last_observation_date': df['date'].max().strftime('%Y-%m-%d'),
        'training_dataframe_csv_sha256': hashlib.sha256(training_df.to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest(),
        'regime_thresholds': {'elevated': float(elevated), 'stress': float(stress)},
        'regime_quantiles': {'elevated': 0.7, 'stress': 0.9},
        'classification_classes': classifier.classes_.tolist(),
        'regression_model': 'Ridge',
        'classification_model': 'Logistic Regression',
        'evaluation_scope': 'Fitted on all labelled rows; not the annual walk-forward models.',
        'python_version': platform.python_version(),
        'package_versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'source_sha256': {
            path.relative_to(ROOT_PATH).as_posix(): hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for path in sorted((ROOT_PATH / 'src' / 'market_risk').rglob('*.py'))
        },
    }
    save_artifacts(ridge, classifier, metadata, output_dir)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ARTIFACTS_PATH)
    args = parser.parse_args(argv)
    metadata = train_and_save(load_model_dataset(args.dataset), args.output_dir)
    print(f"Trained on {metadata['training_rows']} labelled rows through {metadata['training_end']}.")
    print(f'Artifacts and provenance saved to {args.output_dir}')


if __name__ == '__main__':
    main()
