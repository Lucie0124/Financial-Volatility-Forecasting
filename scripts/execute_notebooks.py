"""Execute every notebook in its own fresh kernel; stop on any cell error."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT_PATH = Path(__file__).resolve().parents[1]


def execute_all(output_dir, runtime_dir, paths=None):
    environment = {
        **os.environ,
        'IPYTHONDIR': str(Path(runtime_dir) / 'ipython'),
        'MPLCONFIGDIR': str(Path(runtime_dir) / 'matplotlib'),
    }
    for path in sorted((ROOT_PATH / 'notebooks').glob('*.ipynb')) if paths is None else paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, timeout=180, km=manager, resources={'metadata': {'path': str(path.parent)}})
        try:
            client.execute(env=environment)
        finally:
            if client.kc is not None:
                client.kc.stop_channels()
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        destination = (output_dir / path.name) if output_dir else path
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        print(f'Executed {path.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Save executed copies outside notebooks/.')
    args = parser.parse_args()
    # Use this interpreter and temporary config, independent of user kernels
    # and IPython startup files. Passing a manager requires explicit cleanup.
    with TemporaryDirectory(prefix='market-risk-notebooks-') as runtime_dir:
        execute_all(args.output_dir, runtime_dir)


if __name__ == '__main__':
    main()
