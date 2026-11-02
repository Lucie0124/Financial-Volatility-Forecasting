import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from market_risk.evaluation.run_evaluations import (
    evaluate_all_models, main, save_evaluation_results,
)
from market_risk.modelling.dataset import iter_walk_forward_folds
from market_risk.evaluation.provenance import verify_saved_evaluation


def test_all_evaluators_share_eligible_dates_and_export_verifiable_results(corrected_dataset, tmp_path):
    df = corrected_dataset.copy()
    excluded = pd.Timestamp('2022-02-03')
    df.loc[df['date'].eq(excluded), 'vix'] = np.nan
    tables = evaluate_all_models(df)
    reference = tables['persistence/predictions.csv'][['date', 'y_true']]
    sizes = tables['persistence/fold_results.csv'][['test_year', 'train_size', 'test_size']]
    assert excluded not in reference['date'].tolist()
    for name, filename in [('ridge', 'fold_results.csv'), ('lightgbm', 'results.csv')]:
        pd.testing.assert_frame_equal(tables[f'{name}/predictions.csv'][['date', 'y_true']], reference)
        pd.testing.assert_frame_equal(tables[f'{name}/{filename}'][sizes.columns], sizes)
    classifiers = tables['classification/predictions.csv']
    assert set(classifiers['model']) == {'Dummy', 'Logistic Regression', 'LightGBM'}
    for _, group in classifiers.groupby('model'):
        pd.testing.assert_frame_equal(group[['date', 'y_true']].reset_index(drop=True), reference)
    for _, group in tables['classification/fold_results.csv'].groupby('model'):
        pd.testing.assert_frame_equal(group[sizes.columns].reset_index(drop=True), sizes)

    save_evaluation_results(tables, df, tmp_path)
    verify_saved_evaluation(df, tmp_path)
    manifest = json.loads((tmp_path / 'evaluation_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['test_years'] == [2022, 2023]
    assert len(manifest['output_sha256']) == 16
    for filename, checksum in manifest['output_sha256'].items():
        path = tmp_path / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        assert not any(column.startswith('Unnamed:') for column in pd.read_csv(path).columns)
    with pytest.raises(ValueError, match='different data'):
        verify_saved_evaluation(df.iloc[:-1], tmp_path)
    with (tmp_path / 'ridge/predictions.csv').open('a') as file:
        file.write('\n')
    with pytest.raises(ValueError, match='result changed'):
        verify_saved_evaluation(df, tmp_path)


def test_cli_rejects_legacy_labels_before_writing(corrected_dataset, tmp_path):
    df = corrected_dataset.copy()
    df['target_volatility_5d'] *= np.sqrt(5)
    path = tmp_path / 'legacy.csv'
    df.to_csv(path, index=False)
    output = tmp_path / 'results'
    with pytest.raises(ValueError, match='Regenerate'):
        main(['--dataset', str(path), '--output-dir', str(output)])
    assert not output.exists()


def test_interrupted_export_invalidates_previous_completion_marker(corrected_dataset, tmp_path, monkeypatch):
    marker = tmp_path / 'evaluation_manifest.json'
    marker.write_text('{"status":"complete"}')

    def fail_write(*args, **kwargs):
        raise OSError('interrupted export')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail_write)
    with pytest.raises(OSError, match='interrupted'):
        save_evaluation_results({'ridge/predictions.csv': pd.DataFrame()}, corrected_dataset, tmp_path)
    assert not marker.exists()


def test_fold_generator_rejects_duplicate_or_unsorted_dates(corrected_dataset):
    with pytest.raises(ValueError, match='chronologically'):
        list(iter_walk_forward_folds(corrected_dataset.iloc[::-1]))
    duplicate = corrected_dataset.copy()
    duplicate.loc[1, 'date'] = duplicate.loc[0, 'date']
    with pytest.raises(ValueError, match='unique'):
        list(iter_walk_forward_folds(duplicate))
    with pytest.raises(ValueError, match='no complete'):
        list(iter_walk_forward_folds(corrected_dataset, [2030]))
