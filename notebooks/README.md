# Complete #2756 notebook

`TrustGuard_2756_Complete.ipynb` is the preserved, executed training and analysis companion for historical public submission #2756. Its saved outputs and the adjacent HTML are historical evidence, not results produced by this cleanup. The notebook is copied byte-for-byte; the runner changes execution flags only in memory.

## Run with Python 3.12

Create a separate environment and install this folder's requirements. These PowerShell commands work from any current directory after setting the absolute repository path:

```powershell
$repo = 'C:\path\to\DLW-Datathon-2026'
py -3.12 -m venv "$repo\.venv-notebook"
$python = "$repo\.venv-notebook\Scripts\python.exe"
& $python -m pip install -r "$repo\notebooks\requirements.txt"
& $python "$repo\notebooks\run_notebook.py"
```

On other platforms, create the environment with `python3.12 -m venv` and use its `bin/python`. Pass absolute paths to the same requirements and runner. The runner locates its inputs relative to itself and starts the kernel in `notebooks/`, using the same interpreter that launched it.

The default run uses CPU and skips only the optional CatBoost GPU-training demonstration. It performs fresh contextual cross-validation and full contextual/TAN reconstruction, while retaining the original CatBoost component for exact release reproduction. It is a training workload, not a quick prediction-only check.

Optional flags:

- `--saved-cv`: recompute validation summaries from the saved OOF scores instead of retraining the five contextual folds. Full-data reconstruction still trains.
- `--gpu-demo`: enable the optional historical GPU CatBoost training demonstration; this requires a compatible available GPU. That newly fitted demonstration model is not used in the #2756 reconstruction.

Directly executing the untouched notebook uses its historical flags, including the enabled GPU demonstration. Use the runner for the portable CPU default.

## Inputs and new results

`inputs/manifest.json` binds the **17 copied inputs** by SHA256. They include training/test data, the frozen #2756 model and predictions, development/repeated-validation evidence and source snapshots. Keep these inputs and the manifest unchanged. This is a reproducibility bundle; the historical public receipt does not establish a new public score.

The runner writes the executed copy to **`outputs/TrustGuard_2756_Complete.executed.ipynb`**, never over the preserved notebook. It clears historical cell outputs in that in-memory copy before execution, saves partial diagnostics on failure, and shuts down its kernel in `finally`. It also writes/refreshes the notebook's ordinary products under `outputs/`, including plots, prediction CSVs, `training_reconstruction.pkl` and `verification_receipt.json`. Successive runs overwrite these generated files; copy aside any run you need to retain. Copied plots and receipts already present in `outputs/` are historical until replaced by a successful new run; a partial run can leave a mixture of old and new auxiliary files.

After a successful complete run, verify the produced reconstruction in a separate CPU process:

```powershell
& $python "$repo\notebooks\verify_fresh_process.py"
```

That verifier requires **`outputs/training_reconstruction.pkl` from the completed run**. A copied receipt alone is insufficient. It blocks fitting and Python network connections, then checks shuffled no-ID predictions against the frozen submission. It is stricter about exact parity than the notebook's reconstruction tolerance, so inspect a failure rather than silently loosening it.

## Separate prediction package

The repository's `track2/` folder holds the separate prediction-only release (`prediction.ipynb`, `model.pkl`, requirements and submission). Use that package for inference; this complete notebook is the training/analysis companion. Running either locally does not upload anything or establish organizer sandbox acceptance.

## Cleanup verification, 5 October 2026

The CPU runner completed all 20 code cells with `--saved-cv`: it recomputed saved validation summaries and freshly reconstructed the full contextual/LR/TAN model. The optional GPU demonstration and fresh cross-validation were disabled. The separate fresh-process verifier reproduced all 12,000 shuffled, no-ID #2756 scores exactly, with fitting and Python network connections blocked. The nine figures and two receipts in `outputs/` reflect this cleanup run; the main notebook and HTML retain their original executed contents. Extra generated models, predictions and the executed copy were moved to the local cleanup archive so they are not included in the final repository.
