"""Verify frozen #2756 hashes and CPU inference; never train or modify the release."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
from predict import ROOT, run_prediction

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert sys.version_info[:2] == (3, 12), 'Use Python 3.12.'
    pins = {'numpy':'2.1.3', 'pandas':'2.2.3', 'scipy':'1.14.1',
            'scikit-learn':'1.5.2', 'catboost':'1.2.10'}
    assert all(version(name) == pin for name, pin in pins.items())
    manifest = json.loads((ROOT/'track2/release_manifest.json').read_text())
    assert manifest['submission'] == 2756
    observed = {name: sha(ROOT/'track2'/name) for name in manifest['files']}
    assert observed == manifest['files'], 'Frozen package hash mismatch.'
    inputs = ROOT/'notebooks/inputs'
    copied = json.loads((inputs/'manifest.json').read_text())
    assert all(sha(inputs/name) == item['sha256'] for name, item in copied.items())
    test = pd.read_csv(inputs/'test.csv', dtype={'id':str})
    expected = pd.read_csv(ROOT/'track2/submission.csv', dtype={'id':str}, float_precision='round_trip')
    assert list(expected.columns) == ['id','prediction']
    assert len(test) == len(expected) == 12000 and test['id'].is_unique
    assert test['id'].equals(expected['id'])
    order = np.random.default_rng(2756).permutation(len(test))
    shuffled = test.iloc[order].drop(columns='id')
    with tempfile.TemporaryDirectory(prefix='trustguard-check-') as scratch:
        scratch = Path(scratch)
        shuffled.to_csv(scratch/'input.csv', index=False)
        namespace = run_prediction(scratch/'input.csv', scratch/'output.csv')
        actual_frame = pd.read_csv(scratch/'output.csv', float_precision='round_trip')
        assert list(actual_frame.columns) == ['prediction']
        actual = actual_frame['prediction'].to_numpy()
        assert len(actual) == 12000 and np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
        delta = float(np.max(np.abs(actual-expected['prediction'].to_numpy()[order])))
        assert delta <= 1e-12, delta
        predict = namespace['predict_fraud']
        sample = shuffled.iloc[:31]
        whole = predict(sample)
        singles = np.array([predict(sample.iloc[[i]])[0] for i in [0,7,30]])
        np.testing.assert_allclose(singles, whole[[0,7,30]], atol=1e-12, rtol=0)
        batch = np.concatenate([predict(sample.iloc[:11]), predict(sample.iloc[11:])])
        np.testing.assert_allclose(batch, whole, atol=1e-12, rtol=0)
    assert {name:sha(ROOT/'track2'/name) for name in observed} == observed
    receipt = {'passed':True, 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'submission':2756, 'rows':len(actual), 'max_absolute_difference':delta,
        'CPU_only':True, 'shuffled_no_ID':True, 'singleton_batch_checks':True,
        'copied_input_hashes_verified':True, 'release_unchanged':True, 'files':observed,
        'python':'.'.join(map(str,sys.version_info[:3])), 'versions':pins,
        'scope':'Code-cell execution of the unchanged prediction notebook; local check, not organizer execution.'}
    (ROOT/'checks').mkdir(exist_ok=True)
    (ROOT/'checks/submission_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main()
