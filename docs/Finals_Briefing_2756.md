# Submission #2756: Finals Briefing

Prepared 4 October 2026 for Team AI of Spades, DLW Datathon Track 2: TrustGuard financial fraud detection.

This briefing explains the exact `upload_2756.zip` supplied in this chat. The ZIP, its extracted files, the two reports, current model state, source protocols and saved evaluation records were inspected. It is a companion briefing, not a replacement submission or technical report. The user's statement that the team reached finals is accepted; earlier workspace notes about eligibility are historical.

## 1. What You Should Be Able to Say Immediately

We rank transactions by fraud risk so investigators can spend their limited attention on the most suspicious transactions. Only 353 of 20,000 training transactions are fraudulent, so a system that calls every transaction legitimate gets 98.235% accuracy while catching no fraud. Our submitted model combines a strongly regularized logistic regression, CatBoost and a small Bayesian dependency model. Its distinguishing feature is a comparison between the observed transaction and the typical legitimate transaction under similar available context. The final output is a continuous risk score, not an automatic block decision.

The exact blend is:

`risk = 0.63 * contextual_logistic + 0.27 * CatBoost + 0.10 * TAN`

The saved public result for submission #2756 is **0.18900 PR-AUC**. Its five-fold development evaluation has **0.2364935 average precision**. Across five repeated five-fold evaluations, the mean of the five pooled out-of-fold AP values is **0.2330231**. These are three different pieces of evidence, not interchangeable measures of future performance.

The most useful operational example: ranking all 20,000 development transactions by their out-of-fold scores, the top 200 contain **84 frauds**. Reviewing 1% of transactions captures **23.8% of fraud**, with **42% precision** and **116 legitimate transactions in that queue**. This is a development-data estimate, not a measured deployment result.

## 2. Two Issues to Understand Before Presenting

### The ZIP contains two different technical reports

| File | What it actually describes |
|---|---|
| `technical_report.pdf` | Historical submission #370: 70% LR and 30% CatBoost; public score 0.18231 |
| `technical_report_2756.pdf` | Current #2756: 63% contextual LR, 27% CatBoost and 10% TAN; public score 0.18900 |

Both are one-page PDFs. The model and prediction notebook implement #2756. Use the #2756 report to explain the model. This inspection does not establish which PDF the organizers accepted; the ZIP's filenames alone cannot establish that.

### Public improvement and repeated local evidence disagree

#2756 improved the recorded public score from #2510's **0.18776** to **0.18900**, a gain of **0.00124**. However, repeated paired evaluation gave #2510 a mean pooled AP of **0.2339413**, versus **0.2330231** for #2756. #2756 won **1 of 5 repeats** and **8 of 25 folds**.

Say: "This was our retained public-scoring release. The contextual addition gave mixed local evidence: a small five-fold improvement, but a slightly worse repeated mean. We do not claim a consistently established gain from that addition."

Do not say that repeated validation proved #2756 better, that the extra features caused its public gain, or that a higher public score proves better hidden-test performance.

## 3. The Data and Its Meaning

The current local datasets were checked directly:

| Item | Verified value |
|---|---:|
| Training transactions | 20,000 |
| Fraudulent transactions | 353 |
| Legitimate transactions | 19,647 |
| Fraud prevalence | 1.765% |
| Testing transactions | 12,000; labels unavailable |
| Raw predictors | 10 |
| Training rows with any missing predictor | 975, or 4.875% |
| Test rows with any missing predictor | 1,663, or 13.858% |
| Mean transaction amount, train / test | 252.664 / 523.930 |
| Duplicate training predictor rows | 0 |

The ten predictors are:

| Predictor | Available information and boundary |
|---|---|
| `transaction_amount` | Current transaction amount; currency semantics were not established here |
| `transaction_hour` | Hour of day, not a full date or timestamp |
| `merchant_category` | Category, not a unique merchant identifier |
| `country` | A country code; it does not identify the customer's home country |
| `transaction_channel` | Channel such as ecommerce or card-present |
| `transactions_last_24h` | Supplied count; precise window inclusion semantics remain undocumented |
| `spend_last_24h` | Supplied aggregate spend; cannot assume it excludes the current transaction |
| `account_age` | Numeric account age; the supplied unit was not verified |
| `new_device` | Binary risk signal; it does not itself prove account takeover |
| `transactions_last_1h` | Supplied recent count; the 1h and 24h windows can overlap |

`id` identifies a transaction row and is excluded from prediction. `fraud` is the supervised training label and is absent from test input. There is no customer ID, device fingerprint or full timestamp from which to build reliable longitudinal histories.

Do not drop incomplete rows or discard large amounts just because they look like statistical outliers. Both may contain the transactions of interest. This submission retains rows and handles missing predictors within each branch.

The observed shift is substantial: test transactions are larger on average and are more likely to have missing fields. This is evidence that input distributions differ. It does not identify the hidden fraud rate, prove a changed fraud mechanism, or fully explain the CV/public score gap.

## 4. Feature Engineering: What Goes into the Models

### Base logistic features

The saved preprocessing produces **43 base columns**: **28 categorical indicator columns** and **15 numeric columns**. The contextual branch appends **two residual columns**, making **45 classifier inputs**.

Numeric processing uses training-fitted median imputation and scaling. Categories have an explicit missing sentinel; unseen categories are ignored by the one-hot encoder rather than forcing new columns. Indicator columns are not standardized as if they were continuous variables.

Useful engineered signals include:

- Log transforms of transaction amount, recent spend and account age, which reduce the effect of large numeric ranges.
- Amount relative to recent average spend: `log1p(amount / (spend_last_24h / transactions_last_24h + 1))`, with the pipeline's numeric safeguards and missing-value handling. A zero 24-hour count is first replaced by a missing value; the resulting missing ratio is median-imputed. The `+1` does not make zero-count division valid. It is a contextual aggregate ratio, not a verified personal spending baseline.
- A night flag for hours **23, 0, 1, 2 and 3**.
- A non-SG flag for an observed country code different from `SG`. Describe it as "non-SG", not necessarily "foreign to this customer".
- Five explicit interactions: new device with age below 90; new device with log amount; age below 90 with log amount; new device with last-hour count; and night with new device.

An interaction lets the effect of one predictor depend on another. For example, a new device combined with a young account can behave differently from either factor alone. These are predictive associations, not evidence that a device or age causes fraud.

Use "account age below 90 in the supplied units" unless the organizers confirm the unit. The newer PDF's phrase "90 days" is stronger than the verified data description.

### Contextual median residuals

This is the most distinctive mechanism to understand. The model asks two questions:

1. For legitimate transactions with similar available context, what is the typical **log transaction amount**?
2. Under that context, what is the typical **log last-hour transaction count**?

Two conditional median predictors are fitted using legitimate training transactions with observed targets. Each predictor excludes its own target from its inputs. Context includes categorical fields, other log-transformed activity fields, new-device status and sine/cosine hour features.

The deployed median predictors are histogram gradient boosting regressors using quantile loss at **0.5**: 100 iterations, learning rate 0.05, maximum depth 3, at most 7 leaves, minimum leaf size 150 and L2 strength 10. The saved bundle also contains 0.9-quantile predictors, but the submitted notebook uses **only the median predictors**.

The initial residual is `observed log value - predicted legitimate median`. A hypothetical amount of 400 compared with a contextual median of 100 produces a positive log residual; this illustration is not an actual measured fraud score. A positive deviation is not automatically fraud. The supervised classifier learns how useful each deviation is alongside the other features.

In the saved final LR, both residual coefficients are negative after conditioning on the base features. Their effective contributions per one standardized residual unit are approximately -0.00098 for amount and -0.03412 for hourly count in the LR log-odds. This makes it especially important to describe them as **learned contextual adjustments**, not a rule that larger anomalies always increase risk.

The residuals are imputed, have their training-fitted linear component explained by the base design removed, and are standardized. They are then multiplied by **0.2** before entering the LR. With L2 regularization, this smaller feature scale makes large effects from the new residuals more expensive, limiting their influence.

The context is drawn from the legitimate **population** under similar observed fields. It is not the individual customer's historical normal behavior, because the data do not identify customers.

### Why cross-fitting matters

If a legitimate training row helped fit the model that defined that same row's "normality", its residual could be artificially small. The training residuals instead use **three-fold inner cross-fitting**: each training row receives a normal-behavior prediction from donor models that did not train on that row.

During outer CV, those inner donor models and learned transformations stay inside the outer training partition. When exporting the final trained bundle, legitimate reference models are refitted on eligible full-training rows for future inference. The supervised fraud classifier still trains on the saved cross-fitted training residuals.

## 5. The Three Prediction Branches

| Branch | Share | Mechanism | Why it is included |
|---|---:|---|---|
| Contextual logistic regression | 63% | L2-regularized LR, `C=0.1`, unweighted, 45 inputs | A constrained main model with targeted interactions and contextual deviations |
| CatBoost | 27% | 600 depth-5 boosted trees; learning rate about 0.04265; L2 leaf strength about 29.77 | Nonlinear thresholds and categorical combinations |
| Tree-augmented naive Bayes, TAN | 10% | Discretized class-conditional probability model with a tree of predictor dependencies | A different representation of dependency and missing evidence |

### Logistic regression

It maps a weighted feature sum to a probability through a sigmoid. It is linear in transformed **log odds**, not in raw fraud probabilities. Explicit interaction terms extend what it can represent. `C=0.1` specifies stronger regularization than `C=1`; it does not mean a 10% confidence level. [Official scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

The project found regularized logistic models useful across its tested comparisons. That does not prove the true fraud mechanism is logistic or multiplicative, or that sophisticated models cannot do better.

### CatBoost

This is a gradient-boosted tree classifier designed to work with categorical features. Its training uses ordered categorical statistics under the algorithm's documented rules. [Official categorical feature documentation](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic).

The retained CatBoost uses **16 base engineered inputs**, including three categorical inputs. It does not receive the five extra LR interaction columns or the two contextual residuals. It was trained using a GPU; prediction uses the saved model on CPU. No GPU or retraining is required to execute the prediction notebook.

### TAN

Naive Bayes estimates class probabilities by assuming predictor independence conditional on the class. TAN relaxes that assumption: a predictor can depend on one other predictor as well as the fraud class, forming a tree.

This saved TAN uses mostly four numeric bins, with two effective bins for hourly count, and smoothing strength **80**. Its feature tree has hour as its root and nine feature-to-feature edges. Smoothing prevents sparse conditional counts from producing brittle zero probabilities. Its prediction code computes class evidence by passing messages through the tree. All ten saved fields use missing-value marginalization: unknown or unobserved inputs are integrated out instead of assigning an arbitrary single numeric value. This does not establish that missingness is harmless or that the mechanism is missing at random.

### Blending

The notebook averages the three predicted probabilities with fixed weights. It does not multiply probabilities, average logits, or train a new stacking model during inference. The 63/27/10 weights derive from retaining 90% of a 70/30 LR/CatBoost mixture and assigning 10% to TAN.

The fraud classifiers were trained with their respective likelihood/log-loss objectives and selected using AP comparisons. "Optimized for PR-AUC" refers to development selection; this submitted LR is not trained by directly maximizing the nondifferentiable AP statistic.

## 6. Performance: Memorize the Correct Numbers

### Evaluation values

| Evidence | Value | What it means |
|---|---:|---|
| Constant-score AP on the training labels | 0.01765 | The class prevalence baseline |
| #2756 five-fold pooled OOF AP | 0.23649349 | Scores for 20,000 development rows held out from their fold models |
| Same scores, trapezoidal PR area | 0.23552841 | A different PR integration convention |
| Same scores, ROC-AUC | 0.78434352 | Another ranking summary, not the main local selection measure |
| Contextual LR alone, same five-fold scores | 0.23490558 | Constituent development AP |
| CatBoost alone, same five-fold scores | 0.20588848 | Constituent development AP |
| TAN alone, same five-fold scores | 0.14206961 | Constituent development AP |
| Mean pooled AP over five repeated five-fold evaluations | 0.23302312 | Mean of five whole-dataset OOF AP values, not the mean of 25 fold APs |
| Paired previous #2510 repeated mean AP | 0.23394129 | Slightly better repeated local performance than #2756 |
| Recorded #2756 public leaderboard PR-AUC | 0.18900 | One observed scored submission; does not establish private-test performance |

Average precision weights precision by increments in recall. It differs from trapezoidal integration of a PR curve. Therefore label the local number **AP**, and the portal number **public PR-AUC** unless its precise implementation is confirmed. [Official average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

The blend exceeds each constituent on this matched development split. That is evidence of useful complementarity on that split, not proof that every blend component improves every validation scenario or that all other model families are inferior.

**0.18900 is not 18.9% accuracy, 18.9% recall, or 18.9% precision at a chosen threshold.** AP aggregates ranking behavior across operating points.

### Review-budget results

These were recalculated from the saved #2756 confirmation OOF scores with dataset ID alignment checked. They apply to the 20,000 labeled development rows.

| Review budget | Transactions reviewed | Fraud caught | Legitimate alerts | Precision | Fraud recall |
|---|---:|---:|---:|---:|---:|
| Top 0.5% | 100 | 61 | 39 | 61.0% | 17.28% |
| Top 1% | 200 | 84 | 116 | 42.0% | 23.80% |
| Top 5% | 1,000 | 141 | 859 | 14.1% | 39.94% |

The top-100 precision is about **34.6 times the overall training fraud prevalence**. This is precision lift on the development evaluation, not a measured revenue uplift or private-test guarantee.

At the exact development F1-maximizing threshold **0.19376658833462837**, the confusion counts are **TP 83, FP 105, FN 270, TN 19,542**, with precision **44.15%**, recall **23.51%** and F1 **0.30684**. Applying the rounded threshold `0.194` produces different counts. Do not promise exact counts at a rounded threshold, and do not present a threshold selected on these labels as an independently validated production policy.

The submitted notebook applies **no threshold**. It outputs scores. Any review, OTP or blocking policy is an operational layer to be designed and evaluated separately.

## 7. What Validation Establishes and What It Does Not

The development work includes stratified folds, training-only learned preprocessing, legitimate-only donor checks, inner cross-fitting, paired comparisons, slice metrics and artifact replay. These controls reduce direct leakage and help locate implementation mistakes.

There were only **353 fraud examples**. Reusing them across feature discovery, model comparison, new split seeds, thresholds and repeated public submissions can make selected results optimistic. A fresh seed rearranges existing observations; it does not create an untouched dataset. The 25 folds also overlap in their training rows. Their variability cannot be treated as 25 independent experiments.

The contextual recipe originally **failed** its source selection rule: pooled AP changed by **-0.000164973**, despite positive mean-fold change and 3/3 fold wins. A separately written post-source follow-up tested the already observed recipe on five folds. It gained **+0.000156643 AP** over #2510 and won 4/5 folds. That follow-up rule was frozen before its run, but the recipe had already been seen. The later repeated comparison was unfavorable on average.

Consequently, the newer report's "held-out split" should be explained as **out-of-fold development predictions**, and its rule-writing statement should not imply that the whole history was independent prospective confirmation. Training-boundary correctness and freedom from selection bias are different questions.

The public score gap also remains unresolved. Input shift, small positive counts and adaptive selection are plausible contributors, but the available evidence does not measure their individual effects. Hidden labels, fraud prevalence and the final ranking are unknown here.

## 8. What the Submitted Notebook Actually Does

`prediction.ipynb` has one short explanatory markdown cell and one prediction code cell. It is a deployment notebook, not a full EDA/training presentation.

Execution proceeds as follows:

1. Load `model.pkl` from the current working directory.
2. Read a CSV from `DATATHON_INPUT_PATH`, or default to `test.csv`.
3. Select the ten saved feature columns; IDs and any label are not used.
4. Obtain two legitimate-context median predictions and construct the trained residual feature representation.
5. Predict LR, CatBoost and TAN scores and calculate the 63/27/10 blend.
6. Verify output length, finite values and the [0, 1] range.
7. Write one `prediction` column to `DATATHON_OUTPUT_PATH`, or default to `predictions.csv`, preserving input row order.

This differs from the leaderboard CSV, which uses `id,prediction`. The ZIP does not include that leaderboard CSV, the raw training data, source training code or validation results. The model contains trained preprocessing and estimator state, so inference does not need them.

Required runtime for the recorded compatibility checks: Python **3.12**, NumPy **2.1.3**, pandas **2.2.3**, SciPy **1.14.1**, scikit-learn **1.5.2**, and CatBoost **1.2.10**. The ZIP's `requirements.txt` lists only CatBoost, relying on the documented organizer base environment for the other dependencies. It is not a complete generic standalone environment specification.

The model pickle is **822,551 bytes**, about 0.82 MB in decimal units. That size does not measure process memory or latency. The notebook assumes all required columns exist and that its working directory contains `model.pkl`; it is not a general production input-validation service.

Recorded Linux replay checked 12,000 shuffled ID-free rows, environment-path and default-path modes, single-row and 37-row batches. Predictions differed from the saved CSV by at most **1.11e-16**. Fits and Python socket connections were blocked during that recorded check. This is a local compatibility receipt, not proof of organizer execution or OS-level network isolation.

A fresh check during this inspection executed the exact archived notebook in a new Python 3.12 CPU kernel on all 12,000 shuffled ID-free rows. It produced one `prediction` column, all finite values in [0, 1], and **maximum difference 0.0** versus the canonical #2756 CSV after alignment. Selected single-row predictions differed from full-batch scores by at most **6.94e-18**; subset and permutation checks matched exactly. All package hashes remained unchanged. The prediction checks completed successfully; the surrounding verifier then hit a Windows temporary-directory cleanup error, which does not affect those prediction comparisons. Elapsed time including kernel startup was about 44 seconds, not a pure inference latency benchmark.

The locally saved primary organizer prediction template explicitly confirms no internet, no training data, prediction only, shuffled ID-free input, output row-order preservation, one `prediction` column and Track 2 scores in [0, 1]. Its file is stored under the Track 1 official-download folder but explicitly documents both tracks; its Track 2 clauses are applicable here.

## 9. How to Explain the Limitations

### The model still misses most fraud at small review budgets

The 1% review queue catches 84 of 353 frauds and misses 269. This makes it useful as a prioritization mechanism, while leaving a substantial detection gap. Do not frame 42% precision as sufficient justification to block every flagged customer automatically.

### Known-device fraud is difficult

The confirmation slice contains **218 known-device frauds** and **135 new-device frauds**. Known-device AP is about **0.049**, versus about **0.512** for the new-device slice. Slice prevalences differ, so AP values should be interpreted with their own baselines. The important practical point is that many fraudulent transactions lack the conspicuous new-device signal. The dataset cannot support reliable customer-specific sequences.

### Calibrated probability and financial savings are unverified

The outputs are probability estimates, but the evidence does not establish that a score of 0.20 corresponds to a 20% fraud rate in deployment. No calibration layer appears in the submitted notebook. The available data also do not establish fraud losses, recovery rates, investigator costs or OTP friction, so there is no verified dollar-saving claim.

### Input semantics and fairness need further evaluation

The supplied activity windows and age units have unverified semantics. Production features must be available before the scoring decision. A non-SG flag reflects the training reference, not an inherent judgment of people or countries. Country/channel/merchant subgroup false-positive rates and calibration should be evaluated under a real operational policy.

### More data is an opportunity, not a guaranteed solution

Useful next data include customer-level history, reliable event timestamps, device identifiers, merchant identifiers and delayed confirmed fraud outcomes. An appropriate next evaluation would separate tuning from untouched future-time testing, hold out customer groups where relevant, and track recall at fixed review capacity. More data and richer models still require prospective verification.

## 10. A 60-Second Pitch

"We built a transaction risk-ranking system for a setting where only 1.8% of transactions are fraudulent. Calling everything legitimate would achieve 98.2% accuracy and catch nothing, so we focused on precision-recall ranking and the investigator's review budget. Our model combines regularized logistic regression, CatBoost and a Bayesian dependency model. We added two contextual features that compare an amount and its recent transaction activity with what is typical among legitimate transactions under similar available context. In one five-fold development evaluation, the top 1% of transactions contained 84 of the 353 frauds: about 24% recall at 42% precision. Our recorded public score was 0.189 PR-AUC. We also found that local improvements did not always transfer, so we distinguish development results from public results and do not claim verified production performance. The intended use is to prioritize review or additional verification, with thresholds governed by capacity and customer impact."

## 11. Judge Questions and Defensible Answers

**Why does this problem matter?** Fraud is rare, while legitimate customers vastly outnumber fraudsters. A useful system concentrates fraud into a manageable review queue and measures the legitimate-customer burden of doing so.

**Why AP instead of accuracy?** Accuracy is dominated by legitimate rows. AP evaluates how effectively the ranking concentrates fraud near the top across recall levels. ROC-AUC remains useful, but the alert queue's precision and recall communicate the operational tradeoff more directly.

**Why use logistic regression as the main model?** It was a competitive, constrained baseline in this project's tested development comparisons. With few fraud labels, strong regularization and targeted interactions were useful. Its success does not prove that the true mechanism is logistic.

**What is original about the solution?** The project-specific contribution is the feature and validation design: explicit risk-factor interactions, two legitimate-context deviation features with cross-fitted donor predictions, and blending models with different assumptions. LR, CatBoost, TAN and contextual anomaly features are established methods; do not claim to have invented them.

**How is the context model different from a global outlier detector?** It estimates typical amount/activity conditional on available context among legitimate transactions. An unusual value is interpreted relative to that context and then judged by the supervised fraud model; a large raw amount alone need not be suspicious.

**Do you know the customer's normal spending?** No. There is no customer identifier. The reference is conditional population behavior, not personal history.

**Why the 63/27/10 weights?** They retain 90% of the earlier 70/30 LR/CatBoost mixture and add 10% TAN. They are fixed during inference. The evaluation history gives mixed evidence for successive additions, so they are not claimed as mathematically optimal weights.

**How did you address class imbalance?** Stratified splits, AP-based comparisons, review-budget metrics and regularized models. The submitted LR and CatBoost preserve the natural class distribution without class weighting or resampling. The report records unsuccessful weighting/resampling trials; that is a finding about tested recipes, not a general argument against those techniques.

**How did you prevent leakage?** IDs and target labels are excluded from input. Learned transformations stay within training folds. Normality donors are legitimate-only training rows, exclude their prediction target and are cross-fitted for supervised training features. We still acknowledge selection optimism from repeated use of the same labels.

**Is using legitimate labels to build normality leakage?** Using labels inside a training partition is allowed supervised feature construction. Leakage would occur if validation outcomes defined its donor set or its reference model, or if a row inappropriately trained its own reference. The nested donor construction addresses those boundaries.

**How do you handle missing values?** Keep the row. LR uses fitted numeric medians and categorical missing indicators. CatBoost has its saved branch preprocessing. TAN computes probabilities from observed evidence and marginalizes missing or unknown fields. This is robust execution, not proof of robustness to every future missingness mechanism.

**How do you know it is not overfit?** We cannot claim that. Training boundaries and replay checks passed, but repeated label reuse and exploratory selection remain. One contextual comparison improved slightly; its repeated average later worsened. An untouched prospective evaluation is still needed.

**Why did #2756 remain the submitted release if repeated CV favored #2510?** It was retained after a user-approved exploratory public trial improved the recorded score. That is a historical release decision, not proof that repeated CV confirmed it. We report both outcomes and preserve the previous release.

**Why is the public score lower than CV?** We observed input shift and noisy development estimates after extensive adaptive search. We do not have the public labels or an experiment identifying the exact cause. Different PR integration conventions may also affect comparability; they do not establish an explanation for this gap.

**What does TAN add?** A constrained dependency model and a different way to combine observed and missing evidence. Its incremental contribution was not consistently positive under the original local selection rule, even though its exploratory public trial improved the prior release. Diversity alone does not guarantee a gain.

**Can it process one transaction or a shuffled file?** The recorded local replay includes a single row, a small batch and shuffled ID-free input. It preserves row order and uses fitted state. A production service would still require schema checks, monitoring and latency measurement.

**Would you block customers above 0.194?** No production policy was validated at that cutoff. It is a development-selected F1 operating point, and the notebook emits scores without decisions. Review or verification thresholds should follow capacity, cost and prospective outcomes.

**How much money does it save?** The dataset supports fraud-ranking and alert-workload metrics, not a measured financial benefit. That would require loss amounts, investigation cost, recovery rates and a controlled deployment evaluation.

**What would you do next?** Establish a reliable scoring-time feature contract, obtain longitudinal identifiers with suitable access controls, evaluate on untouched future data, calibrate only using separate development data if required, and monitor recall/precision at the chosen review budget.

## 12. A Five-Minute Presentation Flow

1. **0:00-0:40, decision and imbalance.** Explain investigator capacity and the 98.2%-accuracy trap.
2. **0:40-1:25, data and context.** Show ten fields, missingness and the lack of customer history. Explain why amount alone is insufficient.
3. **1:25-2:30, model.** Show the three branches and the 63/27/10 equation. Use a clearly hypothetical contextual-deviation example. Explain cross-fitting in one sentence.
4. **2:30-3:30, evidence.** Present the review-budget table and 0.23649 local AP / 0.18900 recorded public PR-AUC. Label the evaluation data on the table.
5. **3:30-4:15, honesty about the experiment.** State the mixed contextual feature evidence and why repeated CV is not independent testing.
6. **4:15-5:00, use and next step.** Describe review or additional verification, legitimate-alert burden, missing fraud and the prospective evaluation needed.

This is a suggested presentation structure, not a verified official time allowance. Local Oct3 briefing notes say the Jupyter notebook is the finals presentation surface and slides are optional. The exact current invitation, judging weights, slot length and organizer-approved package status were not established in this inspection. The official portal could not be reached during the read-only check.

The supplied prediction notebook has no charts, model comparison or explanation of development results. If using an older training notebook for presentation, label its historical model clearly and show the #2756 formula and metrics from this briefing; it does not retrain the selected release. `Track 2/train_model.py` is also a historical **#1486** builder, not a #2756 reproduction command.

## 13. Phrases to Use Carefully

| Avoid | Say instead |
|---|---|
| "We achieved 98% accuracy" | "An all-legitimate baseline achieves 98.235%, which is why accuracy is unhelpful here" |
| "Our accuracy is 0.189" | "The recorded public PR-AUC is 0.18900" |
| "We detect 42% of fraud" | "At the top-1% review budget, precision is 42% and fraud recall is 23.8% in development OOF predictions" |
| "25 independent tests confirm the improvement" | "Five repeated five-fold evaluations reuse the same labels; their paired mean favored the predecessor" |
| "All selection rules were specified before any results" | "Rules were written for individual experiments, but this contextual follow-up was post-source and exploratory" |
| "New device and age cause fraud" | "Those fields have predictive associations and motivate interaction features" |
| "The model knows each customer's usual behavior" | "It models legitimate population behavior conditional on available context" |
| "Accounts younger than 90 days" | "Account age below 90 in the supplied units, pending unit confirmation" |
| "Missingness is uninformative" | "The chosen methods retained incomplete rows; future missingness risk remains unresolved" |
| "We proved a performance ceiling" | "Our tested configurations did not establish a better result; this is not a bound on what other methods can achieve" |
| "Local replay proves organizer acceptance" | "Local inference reproduced the saved predictions; organizer execution is a separate fact" |

## 14. Evidence and File Identity

The ZIP SHA-256 is `3e16499b4611ace1065290084b145d1e04997c5510029e7b2d3eb568c5a4756d`.

| Archived file | SHA-256 |
|---|---|
| `model.pkl` | `c459d365a97c0dd2ab6f0fc1e664fe2cdfb6170b893968343aec5150f874c57d` |
| `prediction.ipynb` | `3f6addd1abadc0d89ca5a77d8b429dfbe7abee9fcc70094b204de5d4f21177b1` |
| `requirements.txt` | `87e6babba557f4a0d1ac1f6e02c8e4ca6a781eab4526f2b9a38f576a057fd116` |
| `technical_report.pdf` | `061785b95e7ebb4fef9e301443fea8854375ced077fbbb45589b12f70dc2b0ac` |
| `technical_report_2756.pdf` | `534c155102a5a006b40b9165703990e6b4b8c42b54402359d8812acf9bb62150` |

All six archive files match the existing extracted `upload_2756` directory. The model and notebook match the frozen #2756 release. The archive's requirements file contains the same CatBoost pin as the frozen release; the hash differs because LF and CRLF line endings differ, with no dependency change.

Primary local evidence:

- [Exact ZIP](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/DLW Datathon 2026 - FInal Submission/upload_2756.zip>)
- [Current technical report](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/DLW Datathon 2026 - FInal Submission/upload_2756/technical_report_2756.pdf>)
- [Historical report also present in ZIP](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/DLW Datathon 2026 - FInal Submission/upload_2756/technical_report.pdf>)
- [Prediction notebook](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/DLW Datathon 2026 - FInal Submission/upload_2756/prediction.ipynb>)
- [Five-fold result record](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/research/podium_v1/contextual_anomaly/fold_consistent_followup/confirmation_results.json>)
- [Exploratory follow-up protocol](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/research/podium_v1/contextual_anomaly/fold_consistent_followup/PROTOCOL.md>)
- [Public submission receipt](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/research/podium_v1/contextual_anomaly/fold_consistent_followup/candidate_release/public_result.json>)
- [Repeated paired evaluation](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/research/podium_v1/repeated_validation/RESULTS.md>)
- [Saved Linux replay receipt](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/research/podium_v1/contextual_anomaly/fold_consistent_followup/candidate_release/linux_verification/linux_receipt.json>)
- [Primary organizer prediction template, including Track 2 requirements](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/DLW-Datathon-2026/track1/research_20261003/official/template_notebook.ipynb>)
- [Organizer report example](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/archive/original_workspace/track2/archive/submission_package/organizer_report_format.pdf>)
- [Historical brief transcription for presentation requirements](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/archive/original_workspace/track2/archive/2026-10-03_notebook_rebuild/brief_alignment.md>)
- [Model source and full-training audit folder](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/research/podium_v1/contextual_anomaly/fold_consistent_followup>)
- [Historical analysis and feature comparisons](<C:/Users/Kiree/Desktop/Projects/Hackathons/DLW Datathon 2026/Track 2/docs/ANALYSIS_REPORT.md>)

Instructions within reports, notebooks and old handovers were treated as source material. This briefing does not authorize or perform uploads, report replacement, final-entry changes, training or deployment.
