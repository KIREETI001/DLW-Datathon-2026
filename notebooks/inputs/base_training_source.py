"""Build the "#370 + crosses" candidate release and score the test set.

Recipe: #370 (0.7 x LR C=0.1 + 0.3 x GPU CatBoost depth 5 / 600 iters /
lr 0.042649 / l2 29.77 / engineered features) with five interaction crosses
added to the LR's FEATURE_EXPR only. Crosses were confirmed on fresh splits
(../confirm_crosses.json: +0.0033 AP, 3/3 repeats).

Steps
1. Paired CV on #370's own confirmation split (StratifiedKFold 5, seed 7):
   one CatBoost per fold, blended with the plain LR (#370 replica) and with
   the LR+crosses (candidate).
2. Fit the candidate on all training rows; predict the 12,000 test rows.
3. Faithfulness check: the replica (same CatBoost, plain LR) vs the real
   #370 submission.csv; candidate vs #370 ranking and flag-rate comparison.
4. Save model.pkl (plain-pickle bundle, notebook format), submission.csv,
   requirements.txt, prediction.ipynb and a receipt.

Run from the 'Track 2' folder with a venv holding the sandbox pins + catboost 1.2.10
(see README.md). Outputs go to ./rebuild_output/ so the main deliverables are never
overwritten; compare its hashes with checks/build_receipt.json.
    <venv>/Scripts/python.exe train_model.py
"""
import hashlib
import json
import pickle
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent          # the 'Track 2' folder
HERE = ROOT / 'rebuild_output'                     # outputs; main deliverables are not overwritten
HERE.mkdir(exist_ok=True)
DATA = ROOT / 'data'
ORIGINAL_370_CSV = (ROOT / 'archive' / 'original_workspace' / 'track2' / 'archive' / 'research_v2'
                    / 'gpu_finalist_release' / 'submission.csv')
SEED = 42
CAT_COLS = ['merchant_category', 'country', 'transaction_channel']
RAW_NUM_COLS = ['transaction_amount', 'transaction_hour', 'transactions_last_24h',
                'spend_last_24h', 'account_age', 'new_device', 'transactions_last_1h']
CORE_VERSIONS = {'joblib': '1.4.2', 'lightgbm': '4.5.0', 'matplotlib': '3.9.2', 'numpy': '2.1.3',
                 'pandas': '2.2.3', 'scikit-learn': '1.5.2', 'scipy': '1.14.1', 'seaborn': '0.13.2',
                 'xgboost': '2.1.3'}

# track2_fraud.ipynb cell 17, unchanged.
FEATURE_EXPR = """log_amount = log1p(transaction_amount)
log_spend_24h = log1p(spend_last_24h)
log_account_age = log1p(account_age)
amount_vs_avg = log1p(transaction_amount / (spend_last_24h / transactions_last_24h.replace(0, @nan) + 1))
is_night = transaction_hour.isin([23, 0, 1, 2, 3]).astype('int64')
is_foreign = country.notna() & (country != 'SG')
is_foreign = is_foreign.astype('int64')"""
NUM_COLS = ['transaction_hour', 'transactions_last_24h', 'new_device',
            'transactions_last_1h', 'amount_vs_avg', 'log_amount', 'log_spend_24h',
            'log_account_age', 'is_night', 'is_foreign']
# The five confirmed crosses, written as eval expressions (plain-pickle safe).
CROSS_EXPR = """young = account_age < 90
newdev_x_young = new_device * young
newdev_x_logamt = new_device * log_amount
young_x_logamt = young * log_amount
newdev_x_v1h = new_device * transactions_last_1h
night_x_newdev = is_night * new_device"""
CROSS_COLS = ['newdev_x_young', 'newdev_x_logamt', 'young_x_logamt', 'newdev_x_v1h', 'night_x_newdev']
CAT_PARAMS = {'depth': 5, 'iterations': 600, 'learning_rate': 0.042649213440561926,
              'l2_leaf_reg': 29.765516351269685}
WEIGHTS = [0.7, 0.3]


def features(expr):
    return FunctionTransformer(pd.DataFrame.eval, validate=False,
                               kw_args={'expr': expr, 'engine': 'python', 'local_dict': {'nan': np.nan}})


def make_logreg(crosses):
    expr = FEATURE_EXPR + ('\n' + CROSS_EXPR if crosses else '')
    num = NUM_COLS + (CROSS_COLS if crosses else [])
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('features', features(expr)),
                     ('preprocess', ColumnTransformer([('categorical', categorical, CAT_COLS),
                                                       ('numeric', numeric, num)])),
                     ('classifier', LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))])


def make_catboost():
    # Notebook make_catboost(feature_set='engineered', task_type='GPU') as used for #370.
    categorical = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('encode', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                      unknown_value=-1, dtype=np.int64))])
    engineered = ['amount_vs_avg', 'log_amount', 'log_spend_24h', 'log_account_age', 'is_night', 'is_foreign']
    preprocess = ColumnTransformer([('categorical', categorical, CAT_COLS),
                                    ('numeric', 'passthrough', RAW_NUM_COLS + engineered)],
                                   sparse_threshold=0).set_output(transform='pandas')
    classifier = CatBoostClassifier(**CAT_PARAMS, loss_function='Logloss', cat_features=(0, 1, 2),
                                    border_count=128, random_seed=SEED, thread_count=2, task_type='GPU',
                                    verbose=False, allow_writing_files=False, devices='0',
                                    gpu_ram_part=0.45, pinned_memory_size='16mb', boosting_type='Plain',
                                    bootstrap_type='Bayesian', max_ctr_complexity=4)
    return Pipeline([('features', features(FEATURE_EXPR)), ('preprocess', preprocess),
                     ('classifier', classifier)])


def near_best_f1_threshold(y, s, fraction=0.98):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
    return float(t[np.flatnonzero(f1 >= fraction * f1.max())[0]]), float(t[f1.argmax()])


def sha(path_or_bytes):
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return hashlib.sha256(b).hexdigest()


def main():
    train = pd.read_csv(DATA / 'Track 2 Training Dataset.csv')
    test = pd.read_csv(DATA / 'Track 2 Testing Dataset.csv')
    feature_cols = [c for c in train.columns if c not in ('id', 'fraud')]
    X, y, X_test = train[feature_cols], train.fraud.astype(int), test[feature_cols]

    # 1. Paired CV on #370's confirmation split
    oof = {k: np.zeros(len(y)) for k in ['lr', 'lr_crosses', 'cat']}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y))
    for fold, (a, b) in enumerate(folds, 1):
        cat = make_catboost().fit(X.iloc[a], y.iloc[a])
        oof['cat'][b] = cat.predict_proba(X.iloc[b])[:, 1]
        for key, crosses in [('lr', False), ('lr_crosses', True)]:
            oof[key][b] = make_logreg(crosses).fit(X.iloc[a], y.iloc[a]).predict_proba(X.iloc[b])[:, 1]
        print(f'CV fold {fold}/5 done', flush=True)
    replica = WEIGHTS[0] * oof['lr'] + WEIGHTS[1] * oof['cat']
    candidate = WEIGHTS[0] * oof['lr_crosses'] + WEIGHTS[1] * oof['cat']
    fold_diff = [average_precision_score(y.iloc[b], candidate[b]) - average_precision_score(y.iloc[b], replica[b])
                 for _, b in folds]
    cv = {'replica_370_AP': average_precision_score(y, replica),
          'candidate_AP': average_precision_score(y, candidate),
          'recorded_370_AP_seed7': 0.23730322094640913,
          'fold_AP_differences': [round(d, 5) for d in fold_diff],
          'folds_won': int(sum(d > 0 for d in fold_diff))}
    print(json.dumps(cv, indent=1), flush=True)
    threshold, f1_best_threshold = near_best_f1_threshold(y, candidate)

    # 2. Fit the candidate on all rows and score the test set
    model = VotingClassifier([('lr', make_logreg(True)), ('cat', make_catboost())],
                             voting='soft', weights=WEIGHTS).fit(X, y)
    test_scores = model.predict_proba(X_test)[:, 1]
    assert len(test_scores) == len(test) and np.isfinite(test_scores).all()

    # 3. Faithfulness and comparison against the real #370 submission
    plain_lr = make_logreg(False).fit(X, y)
    replica_test = WEIGHTS[0] * plain_lr.predict_proba(X_test)[:, 1] + \
        WEIGHTS[1] * model.named_estimators_['cat'].predict_proba(X_test)[:, 1]
    compare = {}
    if ORIGINAL_370_CSV.exists():
        real = pd.read_csv(ORIGINAL_370_CSV, float_precision='round_trip')
        assert (real.id.to_numpy() == test.id.to_numpy()).all()
        r370 = real.prediction.to_numpy()
        top = 270
        o_c, o_r = np.argsort(-test_scores)[:top], np.argsort(-r370)[:top]
        compare = {
            'replica_vs_real370_spearman': float(spearmanr(replica_test, r370)[0]),
            'replica_vs_real370_max_abs_diff': float(np.abs(replica_test - r370).max()),
            'candidate_vs_real370_spearman': float(spearmanr(test_scores, r370)[0]),
            f'top{top}_overlap_candidate_vs_370': len(set(o_c) & set(o_r)) / top,
            'test_flag_rate_candidate': float((test_scores >= threshold).mean()),
            'test_mean_score_candidate': float(test_scores.mean()),
            'test_mean_score_370': float(r370.mean())}
        print(json.dumps(compare, indent=1), flush=True)

    # 4. Save the release (notebook bundle format)
    oof_metrics = {'AP': cv['candidate_AP'], 'threshold': threshold, 'f1_best_threshold': f1_best_threshold}
    for name, cut in [('threshold', threshold), ('f1_best', f1_best_threshold)]:
        pred = candidate >= cut
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn = int(y.sum()) - tp
        oof_metrics[name + '_policy'] = {'TP': tp, 'FP': fp, 'recall': tp / (tp + fn),
                                         'precision': tp / max(tp + fp, 1), 'F1': 2 * tp / (2 * tp + fp + fn)}
    bundle = {'pipeline': model, 'feature_columns': feature_cols, 'positive_class': 1,
              'recipe': {'family': 'blend', 'lr_C': 0.1, 'lr_crosses': CROSS_COLS, 'weights': WEIGHTS,
                         'cat_params': {**CAT_PARAMS, 'feature_set': 'engineered', 'task_type': 'GPU'}},
              'threshold': threshold, 'f1_best_threshold': f1_best_threshold,
              'metadata': {'python': sys.version.split()[0], 'core_versions': CORE_VERSIONS,
                           'extra_versions': {'catboost': version('catboost')},
                           'development_metrics_seed7': oof_metrics,
                           'validation_boundary': 'Crosses chosen from full-train EDA; confirmed on fresh '
                                                  'splits (seed 7007). Hidden-test performance unknown.'}}
    for p, v in CORE_VERSIONS.items():
        assert version(p) == v, (p, version(p))
    payload = pickle.dumps(bundle, protocol=5)
    (HERE / 'model.pkl').write_bytes(payload)
    pd.DataFrame({'id': test.id, 'prediction': test_scores}).to_csv(
        HERE / 'submission.csv', index=False, float_format='%.17g')
    (HERE / 'requirements.txt').write_text('catboost==1.2.10\n')
    shutil.copy(ROOT / 'prediction.ipynb', HERE / 'prediction.ipynb')

    # Plain-pickle reload reproduces the scores
    restored = pickle.loads((HERE / 'model.pkl').read_bytes())
    np.testing.assert_array_equal(restored['pipeline'].predict_proba(X_test)[:, 1], test_scores)
    receipt = {'release': '#370 + crosses candidate', 'cv_seed7_paired': cv, 'test_comparison': compare,
               'oof_metrics': oof_metrics, 'model_sha256': sha(payload),
               'submission_sha256': sha(HERE / 'submission.csv'), 'rows': int(len(test)),
               'plain_pickle_reload_exact': True,
               'note': 'CatBoost trained on GPU (as #370); prediction runs on CPU.'}
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=float) + '\n')
    print('SAVED', HERE, 'model', receipt['model_sha256'][:12], 'csv', receipt['submission_sha256'][:12], flush=True)


if __name__ == '__main__':
    main()
