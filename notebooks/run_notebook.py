"""Execute a copy of the historical companion; preserve the source notebook."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
PATH = HERE / 'TrustGuard_2756_Complete.ipynb'
OUTPUT = HERE / 'outputs' / 'TrustGuard_2756_Complete.executed.ipynb'


def set_flag(notebook, name, enabled):
    original = f'{name} = True'
    cells = [cell for cell in notebook.cells
             if cell.cell_type == 'code' and original in cell.source]
    if len(cells) != 1 or cells[0].source.count(original) != 1:
        raise ValueError(f'Expected one historical setting: {original}')
    cells[0].source = cells[0].source.replace(original, f'{name} = {enabled}', 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-demo', action='store_true',
                        help='Also train the optional historical CatBoost GPU demo.')
    parser.add_argument('--saved-cv', action='store_true',
                        help='Use saved validation scores; full reconstruction still trains.')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Use Python 3.12 with notebooks/requirements.txt installed.')

    source_hash = hashlib.sha256(PATH.read_bytes()).hexdigest()
    notebook = nbformat.read(PATH, as_version=4)
    set_flag(notebook, 'RUN_CATBOOST_TRAINING_DEMO', args.gpu_demo)
    set_flag(notebook, 'RUN_FRESH_CONTEXT_CV', not args.saved_cv)
    # Partial failures must not retain historical cell outputs as if newly executed.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                               '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=1800,
                            resources={'metadata': {'path': str(HERE)}})
    kernel_env = os.environ.copy()
    if not args.gpu_demo:
        kernel_env['CUDA_VISIBLE_DEVICES'] = '-1'

    def started(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Executing code cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client.on_cell_start = started
    try:
        client.execute(env=kernel_env)
    finally:
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        finally:
            # Write only the executed copy, including useful partial failure output.
            nbformat.write(notebook, OUTPUT)

    errors = [(i, output.get('ename'), output.get('evalue'))
              for i, cell in enumerate(notebook.cells)
              for output in cell.get('outputs', []) if output.output_type == 'error']
    assert not errors, errors
    assert all(cell.execution_count is not None for cell in notebook.cells
               if cell.cell_type == 'code')
    nbformat.validate(notebook)
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == source_hash
    receipt = json.loads((HERE / 'outputs/verification_receipt.json').read_text())
    assert receipt['passed']
    print(json.dumps({'executed_code_cells': sum(c.cell_type == 'code' for c in notebook.cells),
        'errors': errors, 'verification_passed': receipt['passed'],
        'gpu_demo': args.gpu_demo, 'fresh_context_cv': not args.saved_cv,
        'executed_notebook': str(OUTPUT), 'source_notebook_sha256': source_hash,
        'executed_notebook_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
