# Track 2: Finance, Intelligent Financial Fraud Detection

**Task:** Imbalanced binary classification. Predict whether a transaction is fraud (label) while minimising false alarms on legitimate customers.
**Data given:** labelled train set + unlabelled test set (public leaderboard = indicative only; final score is on a hidden private test set).
**Trap:** accuracy is meaningless here. A model that predicts "not fraud" for everything scores about 99% and is useless.

## Key numbers
| Item | Value |
|---|---|
| Round 1 score | Model performance 60% (PR-AUC, F1, Recall) + Technical proposal 40% |
| Deadline | 11:30 AM, Sun 4 Oct 2026 (portal closes) |
| Finalists announced | 2:00-2:30 PM, 4 Oct |
| Finalist registration | by 4:30 PM, at least 2 members present |
| Round 2 | 4:30-6:00 PM, booth showcase, present from the notebook (slides optional) |
| Prizes (each member) | 1st Acer 27" monitor, 2nd Sony earbuds, 3rd powerbank |
| Team size | max 4; one submission per team |

## Dataset numbers (fill in once downloaded)
| Item | Value |
|---|---|
| Train rows x cols | ___ x ___ |
| Test rows x cols | ___ x ___ |
| Label column name | ___ |
| Fraud count / % in train | ___ / ___% |
| Fraud-to-legit ratio (use as `scale_pos_weight`) | 1 : ___ |
| Missing values (% per column) | ___ |
| Categorical columns and cardinality | ___ |
| Amount: mean / median / max (fraud vs legit) | ___ |
| Time column and span | ___ |
| Does the sample submission need 0/1 or probabilities? | ___ |

Print them all:
```python
import pandas as pd
tr, te = pd.read_csv("train.csv"), pd.read_csv("test.csv")
print(tr.shape, te.shape)
print(tr[LABEL].value_counts(), tr[LABEL].mean())
print("scale_pos_weight =", (tr[LABEL]==0).sum()/(tr[LABEL]==1).sum())
print(tr.isna().mean().round(3)); print(tr.nunique())
```

## What to do (in order)
1. **Understand (15 min).** Check the label balance and the exact submission format (0/1 or probability). F1 and Recall need hard labels at a threshold, while PR-AUC needs probabilities. Check the sample file for which is expected.
2. **EDA (45 min).** Fraud rate by hour, amount bucket, merchant or category, location, and by customer. Compare fraud versus legit amount distributions. Look for duplicate rows and ID leakage.
3. **Clean.** Handle missing values, cap extreme amounts, encode high-cardinality categoricals (frequency or target encoding fit inside CV folds only).
4. **Features.** log(amount), hour/day, time since the customer's last transaction, transaction count or velocity over a window, amount vs customer average (z-score), new-merchant or new-location flags, distance from home if coordinates exist.
5. **Baseline.** Logistic regression with `class_weight="balanced"`. Record PR-AUC, F1, Recall.
6. **Models.** LightGBM / XGBoost with `scale_pos_weight`; RandomForest with balanced weights. Compare against the baseline. SMOTE is optional and only on training folds; class weights usually do as well and are safer.
7. **Validate properly.** Stratified KFold (or time split if the test set is later). Never resample before splitting. Never evaluate on accuracy or ROC-AUC alone.
8. **Threshold tuning.** Pick the probability cutoff that maximises F1 on out-of-fold predictions (from the precision-recall curve). Explain the trade-off: higher recall catches more fraud but disrupts more legitimate customers.
9. **Error analysis.** Look at false negatives (missed fraud) and false positives (annoyed customers); show a confusion matrix and cost reasoning.
10. **Package.** Export `model.pkl` with preprocessing inside the pipeline, `requirements.txt`, clean notebook, 1-page PDF.

## 1-page report outline
Why accuracy fails here, imbalance handling and why, features and rationale, model choice and assumptions (judges will ask), results table (PR-AUC / F1 / Recall), chosen threshold and business trade-off, limitations (concept drift, label noise).

## Pre-submit checklist
- [ ] Fresh Colab: Runtime > Restart and Run All, no errors
- [ ] Notebook has loading, EDA, features, training, evaluation, prediction
- [ ] `model.pkl` loads in a new Colab and reproduces predictions
- [ ] Prediction CSV matches the required format (labels vs probabilities, row count)
- [ ] All 4 files uploaded once by the team leader before 11:30 AM
- [ ] Be able to explain every modelling choice in Q&A
