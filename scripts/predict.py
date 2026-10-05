"""Run the frozen prediction notebook without requiring a Jupyter kernel."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def run_prediction(input_path, output_path):
    input_path = Path(input_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError('Input and output must be different files.')
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((ROOT/'track2/prediction.ipynb').read_text(encoding='utf-8'))
    previous_cwd = Path.cwd()
    previous_env = {key: os.environ.get(key) for key in ['DATATHON_INPUT_PATH', 'DATATHON_OUTPUT_PATH']}
    with tempfile.TemporaryDirectory(prefix='trustguard-prediction-') as scratch:
        scratch = Path(scratch)
        shutil.copy2(ROOT/'track2/model.pkl', scratch/'model.pkl')
        os.environ['DATATHON_INPUT_PATH'] = str(input_path)
        os.environ['DATATHON_OUTPUT_PATH'] = str(scratch/'predictions.csv')
        namespace = {'__name__': '__main__'}
        try:
            os.chdir(scratch)
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    source = cell['source']
                    source = ''.join(source) if isinstance(source, list) else source
                    exec(compile(source, f'prediction.ipynb:cell{index+1}', 'exec'), namespace)
            with output_path.open('xb') as handle:
                handle.write((scratch/'predictions.csv').read_bytes())
        finally:
            os.chdir(previous_cwd)
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return namespace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run_prediction(args.input, args.output)

if __name__ == '__main__':
    main()
