# Track 1: Sustainability, Smart Campus Analytics

**Task:** Regression. Predict a campus facility's energy consumption (continuous number) from historical usage, environmental conditions, building info, occupancy and time patterns.
**Data given:** labelled train set + unlabelled test set (public leaderboard = indicative only; final score is on a hidden private test set).
**Golden rule:** best model is not the most complex model.

## Key numbers
| Item | Value |
|---|---|
| Round 1 score | Model performance 60% (RMSE main, plus MAE and R²) + Technical proposal 40% |
| Model-performance split (per slide) | RMSE 60% / other metrics 40% |
| Deadline | 11:30 AM, Sun 4 Oct 2026 (portal closes) |
| Finalists announced | 2:00-2:30 PM, 4 Oct |
| Finalist registration | by 4:30 PM, at least 2 members present |
| Round 2 | 4:30-6:00 PM, booth showcase, present from the notebook (slides optional) |
| Prizes (each member) | 1st Armaggeddon keyboard, 2nd Logitech mouse, 3rd Jisulife fan |
| Team size | max 4; one submission per team |

## Dataset numbers (fill in once downloaded)
| Item | Value |
|---|---|
| Train rows x cols | ___ x ___ |
| Test rows x cols | ___ x ___ |
| Target column name | ___ |
| Target mean / std / min / max | ___ / ___ / ___ / ___ |
| Target skew (log-transform needed if > 1) | ___ |
| Missing values (% per column) | ___ |
| Number of buildings / facilities | ___ |
| Time span and sampling interval | ___ to ___, every ___ |
| Train/test time overlap? (random vs future split) | ___ |
| Submission CSV shape and columns | ___ |

Print them all:
```python
import pandas as pd
tr, te = pd.read_csv("train.csv"), pd.read_csv("test.csv")
print(tr.shape, te.shape); print(tr.dtypes); print(tr.isna().mean().round(3))
print(tr.describe().T); print("skew:", tr[TARGET].skew())
print(set(tr.columns) - set(te.columns))   # should be just the target
```

## What to do (in order)
1. **Understand (15 min).** Read the problem slide; note the target, units, and how the test set is split (random or future dates). The split decides your CV scheme.
2. **EDA (45 min).** Target distribution, consumption by hour / weekday / month, by building, vs temperature and occupancy. Check outliers, zeros, sensor gaps, duplicate timestamps.
3. **Clean.** Impute missing values (time-aware: interpolate or forward-fill within a building), cap or flag outliers, encode categoricals (building type -> one-hot or target encode).
4. **Features (biggest gain).** hour, day-of-week, month, weekend/holiday (include Singapore holidays and NTU exam or semester weeks), cyclical sin/cos of hour and month, lag and rolling-mean usage if lags exist in test, temperature x occupancy interactions, floor area or age per building.
5. **Baseline.** Mean per (building, hour) and plain Linear/Ridge. Record RMSE, MAE, R².
6. **Models.** LightGBM / XGBoost / RandomForest versus the baseline. Try log-target if skewed. Tune lightly (Optuna, 20-30 trials).
7. **Validate properly.** If the test set is in the future, use TimeSeriesSplit, not random KFold. Random KFold with lag features leaks and gives a misleading score. Check public-vs-CV gap to avoid overfitting the leaderboard.
8. **Error analysis (judges' special focus).** Residuals by hour, building and season; find where it fails and say why.
9. **Insights.** Turn predictions into campus actions, e.g. peak-hour load shifting, which buildings waste energy off-hours.
10. **Package.** Export `model.pkl` (include the full preprocessing pipeline so it runs on raw test data), `requirements.txt` (`pip freeze` trimmed), clean notebook, 1-page PDF.

## 1-page report outline
Problem understanding, data prep and EDA findings, feature rationale, model choice and its assumptions (judges will ask), results table (RMSE/MAE/R²), limitations and next steps.

## Pre-submit checklist
- [ ] Fresh Colab: Runtime > Restart and Run All, no errors
- [ ] Notebook has loading, EDA, features, training, evaluation, prediction
- [ ] `model.pkl` loads in a new Colab and reproduces predictions
- [ ] Prediction CSV row count and column names match the sample exactly
- [ ] All 4 files uploaded once by the team leader before 11:30 AM
- [ ] Be able to explain every modelling choice in Q&A
