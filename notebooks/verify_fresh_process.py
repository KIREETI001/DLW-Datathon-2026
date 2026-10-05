"""Prediction-only check of the notebook's exported model in a fresh CPU process."""

import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import ast
import json
import pickle
import socket
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
notebook = nbformat.read(HERE / 'TrustGuard_2756_Complete.ipynb', as_version=4)
for marker in ['def context_target_values', 'def tan_fit_encoder']:
    selected = [cell.source for cell in notebook.cells
                if cell.cell_type == 'code' and marker in cell.source]
    assert len(selected) == 1
    exec(compile(selected[0], marker, 'exec'))
prediction = next(cell.source for cell in notebook.cells
                  if cell.cell_type == 'code' and cell.source.startswith('def predict_components'))
tree = ast.parse(prediction)
functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
                       type_ignores=[])
exec(compile(ast.fix_missing_locations(functions), 'notebook_predict_functions', 'exec'))


def forbidden(*args, **kwargs):
    raise RuntimeError('Fresh inference check forbids training and Python network connections')


for cls in [ColumnTransformer, HistGradientBoostingRegressor, SimpleImputer,
            LogisticRegression, Pipeline, FunctionTransformer, OneHotEncoder,
            StandardScaler, CatBoostClassifier]:
    cls.fit = forbidden
    if hasattr(cls, 'fit_transform'):
        cls.fit_transform = forbidden
socket.socket.connect = forbidden
socket.create_connection = forbidden

model = pickle.loads((HERE / 'outputs/training_reconstruction.pkl').read_bytes())
test = pd.read_csv(HERE / 'inputs/test.csv')
expected = pd.read_csv(HERE / 'inputs/frozen_submission.csv', float_precision='round_trip')
np.testing.assert_array_equal(expected.id.to_numpy(str), test.id.to_numpy(str))
order = np.random.default_rng(52756).permutation(len(test))
with threadpool_limits(2):
    scores = predict_fraud(model, test.iloc[order].drop(columns='id'))
np.testing.assert_array_equal(scores, expected.prediction.to_numpy()[order])
receipt = {'passed': True, 'rows': len(scores), 'max_absolute_difference': 0.0,
           'fresh_process': True, 'CPU_only': True, 'fit_methods_blocked': True,
           'Python_connections_blocked': True, 'shuffled_no_ID': True}
(HERE / 'outputs/fresh_process_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
