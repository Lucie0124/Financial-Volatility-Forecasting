"""Reject stale or incomplete saved evaluations before notebook analysis."""

import hashlib
import json
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]


def verify_saved_evaluation(df, results_dir=ROOT_PATH / 'results'):
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / 'evaluation_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Evaluation is incomplete; run market_risk.evaluation.run_evaluations.')
    data_hash = hashlib.sha256(df.to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest()
    if data_hash != manifest['dataframe_csv_sha256']:
        raise ValueError('Saved evaluation uses different data; regenerate results.')
    for filename, checksum in manifest['source_sha256'].items():
        actual = hashlib.sha256((ROOT_PATH / filename).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
        if actual != checksum:
            raise ValueError(f'Source changed since evaluation: {filename}; regenerate results.')
    for filename, checksum in manifest['output_sha256'].items():
        actual = hashlib.sha256((results_dir / filename).read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f'Saved result changed: {filename}; regenerate results.')
    return manifest


def main():
    from market_risk.modelling.dataset import load_model_dataset

    manifest = verify_saved_evaluation(load_model_dataset())
    print(f"Verified {len(manifest['output_sha256'])} result files, dataset and source hashes.")


if __name__ == '__main__':
    main()
