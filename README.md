# TrustGuard — DLW Datathon 2026

**1st place — Track 2: Intelligent Financial Fraud Detection**

Team **AI of Spades** · DLW Datathon 2026

This repository contains our main submission, **#2756**, and the notebook, report and presentation supporting it. The team won first place in Track 2. The competition result is distinct from the recorded public leaderboard PR-AUC of **0.18900**.

## Submission

| File | Purpose |
|---|---|
| [track2/prediction.ipynb](track2/prediction.ipynb) | Prediction-only notebook; loads the saved model without training |
| [track2/model.pkl](track2/model.pkl) | Exact frozen #2756 model |
| [track2/submission.csv](track2/submission.csv) | Exact 12,000-row `id,prediction` submission |
| [track2/requirements.txt](track2/requirements.txt) | Original sandbox addition: CatBoost 1.2.10 |
| [track2/requirements-local.txt](track2/requirements-local.txt) | Complete pinned local inference dependencies |
| [track2/technical_report.pdf](track2/technical_report.pdf) | Report describing #2756 |
| [track2/release_manifest.json](track2/release_manifest.json) | Original release hashes and recorded public result |

The model combines **63% contextual logistic regression, 27% CatBoost and 10% tree-augmented naive Bayes**. Contextual features compare transaction amount and one-hour activity with estimates from legitimate training transactions. Predictions are continuous fraud-risk scores.

## Run predictions

Use **Python 3.12**. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r track2/requirements-local.txt
.\.venv\Scripts\python.exe scripts/predict.py --input notebooks/inputs/test.csv --output predictions.csv
.\.venv\Scripts\python.exe scripts/verify_submission.py
```

On macOS/Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` for the subsequent commands. GPU access is not required for inference.

The runner executes the code cells from the unchanged prediction notebook in a temporary working directory. It emits the sandbox schema: one `prediction` column in input order. The saved `track2/submission.csv` additionally contains IDs for leaderboard submission. Input IDs are optional for inference; the ten transaction predictors are required. The notebook also supports `DATATHON_INPUT_PATH` and `DATATHON_OUTPUT_PATH`, with defaults `test.csv` and `predictions.csv` when run beside `model.pkl`.

Only load the trusted model supplied with this repository; Python pickle can execute code when loading files.

## Training, analysis and finals

- [Complete executed notebook](notebooks/TrustGuard_2756_Complete.ipynb) and [HTML reading view](notebooks/TrustGuard_2756_Complete.html): EDA, features, training reconstruction, validation and error analysis.
- [Notebook setup and execution](notebooks/README.md): self-contained inputs, CPU execution and an optional GPU training demonstration.
- [Finals presentation](docs/TrustGuard_Finals.pptx), [technical briefing](docs/Finals_Briefing_2756.md) and [Q&A notes](docs/FINALS_QA.md).

The analysis notebook includes the supplied training/test datasets and the frozen #2756 state needed for reproduction. Its training reconstruction retains the original CatBoost weights; a separate optional GPU demonstration is not substituted into the scored model. Generated reconstruction models and predictions stay outside Git.

## Results and limits

| Evaluation | Result |
|---|---:|
| Track 2 competition placing | **1st place** |
| #2756 recorded public PR-AUC | **0.18900** |
| Five-fold development average precision | 0.2364935 |
| Mean of five repeated five-fold pooled AP values | 0.2330231 |
| Fraud captured among the top 200 development scores | 84 of 353 |

The training set has 20,000 transactions and 353 fraud labels; the test set has 12,000 transactions. Development metrics reuse training labels and are not estimates from an untouched external holdout. The contextual addition had mixed repeated-validation results. The public metric and development AP refer to different evaluations; the first-place competition result does not imply a first-place public leaderboard score.

The frozen four-file release is hash-checked by `scripts/verify_submission.py`, which also checks all saved test probabilities, shuffled no-ID inputs, singleton/batched inference and finite scores. [checks/submission_verification.json](checks/submission_verification.json) records the latest local verification. This is separate from any organizer-side execution.

## Repository scope

This is the final Track 2 repository. Earlier experiments, alternate submissions, Track 1 work, old environments and duplicate packages were preserved in a separate local archive during cleanup. They are not required for the main submission and are excluded from the current tree. Existing Git history is retained.
