# TrustGuard — Finals technical Q&A prep (final entry #2756)

Every number below is either recomputed from the data / #2756 out-of-fold scores (4 Oct)
or taken from a saved result in `Track 2/`. Slide numbers refer to the 14-slide deck.
Keep answers to ~20 seconds; point at the slide, then stop talking.

---

## A. Problem and metric

**1. Why PR-AUC and not accuracy or ROC-AUC?** (slide 2)
1.77% fraud: "always legitimate" scores 98.24% accuracy and catches 0 of 353. ROC-AUC
looks good almost by default under heavy imbalance because the false-positive rate is
diluted by 19,647 negatives. PR-AUC measures precision among what we flag, which is what
an investigator experiences (Saito & Rehmsmeier 2015). Random-ranking PR-AUC = the fraud
rate, 0.0177; we reach 0.2365. (Our ROC-AUC, if asked: 0.784.)

**2. Which PR-AUC formula?**
sklearn average precision (step-wise). The portal doesn't specify. We checked six matched
models: AP and trapezoidal PR area gave the same ordering, differing by < 0.001.

**3. Why submit probabilities, not 0/1 labels?**
PR-AUC needs a ranking. The cut-off is an operating decision that depends on review
capacity (slide 10).

## B. Data

**4. What data-quality checks did you do?** (slide 4)
No duplicate rows; IDs unique and excluded from every model; valid ranges. account_age is
capped at 4,000 days (1.77% of rows sit at the cap), amount at 9,000. The 24h aggregates
have undocumented semantics (some rows have an amount larger than the whole 24h spend), so
we treat "amount vs 24h average" as a mathematical feature, not a verified personal baseline.

**5. How did you handle missing values?** (slides 4, 7)
Kept every row. Numeric: median imputation fitted inside each training fold. Categorical:
an explicit "missing" level. TAN integrates over missing inputs exactly. Tested and lost:
missing flags (−0.0019 AP), MICE (−0.0019), KNN (−0.0024).
Train: 4.88% of rows have a gap; test: 13.86%. Incomplete training rows are 2.05% fraud
(20/975) vs 1.75% complete; too few to treat missingness as a strong signal.

**6. How do you know there's no leakage?**
IDs never enter a model. Every imputer, encoder, CatBoost model and the legitimate-only
median models are refit inside each training fold; the context residuals are additionally
cross-fitted inside the training fold so a row never scores itself. No test labels exist.
A six-test ID/row-order screen found no signal (minimum Holm-adjusted p = 0.81).

**7. What about outliers?**
Kept. Fraud is the outlier class, so removing extremes removes signal. Log transforms tame
skew; isolation-forest outlier scores were tested and added nothing (+0.0002).

## C. Features

**8. How did you find the five interaction crosses?** (slides 5, 7)
Multi-factor segment tables. New device × account < 90 days is 18.4% fraud; if risks just
multiplied it would be 8.7%. Logistic regression assumes multiplication, so it under-scores
that corner. Crosses: +0.0056 AP, 3/3 repeats; then confirmed on unused splits inside the
blend, +0.0033, 3/3, against a bar written beforehand (> +0.003, ≥ 2/3).
*Caveat to volunteer:* the segment tables were computed on all training rows, so some
selection optimism remains.

**9. What exactly are the "context residuals"?** (slide 6)
Two median-regression models trained only on legitimate transactions predict log amount
and log last-hour count from the other features. Residual = actual − predicted ("how
unusual is this for a normal customer like this?"). Residuals are de-correlated from the
LR's existing features, standardised, down-weighted (×0.2) and added to the LR.
Local gain was tiny (+0.00016 on confirmation, 4/5 folds); this release gave our best
public score. Don't oversell it.

**10. Why didn't cyclical hour encoding help?** (slide 6)
Hour risk is U-shaped but noisy (~15 frauds per hour). A single night flag captures it;
sin/cos lost 0.0028 AP, six hour-blocks lost 0.0005.

**11. Which features matter most / is it explainable?**
New device, account age, last-hour count, amount and their crosses. LR coefficients are
associations after scaling, not causes. Never say "new device means account takeover".

## D. Models

**12. Why is logistic regression your strongest model?** (slide 8)
Main effects combine roughly additively in log-odds, which is LR's exact functional form;
with only 353 positives, a regularised low-variance model wins. Standalone on identical
folds (earlier feature set): LR 0.228 vs CatBoost 0.219, TabICL 0.215, EBM 0.211,
LightGBM 0.205, XGBoost 0.198, TabM 0.197. Also tried RBF SVM, TabPFN, AutoGluon.

**13. Why a blend, and how were the weights chosen?**
Trees add thresholds/interactions; TAN adds dependency structure and native missing-value
handling. Weights are nested: 0.7 LR / 0.3 CatBoost chosen by CV search (#1486), then
10% TAN (#2510): 0.9×0.7 = 0.63, 0.9×0.3 = 0.27. Component PR-AUC alone: LR 0.2349,
CatBoost 0.2059, TAN 0.1421 → blend 0.2365.
*Honest:* TAN passed its first screen (+0.0025) but only weakly confirmed (+0.0003, 2/5
folds); it was kept as an approved exploratory addition.

**14. What is the TAN component?**
Tree-augmented naive Bayes: features are binned (hour into four 6-hour blocks), each
feature depends on the class plus one parent feature (a tree). Missing inputs are summed
over exactly by message passing, so no imputation is needed.

**15. CatBoost settings?**
Depth 5, 600 trees, strong L2 (~30); trained on GPU, predicts on CPU (CPU replay is exact).

**16. How did you handle class imbalance? SMOTE? Class weights?**
Stratified folds, PR-AUC for selection, natural class prior. Class weighting and resampling
were tested: best resampled forest 0.2342 vs 0.2359 baseline; weighted LR failed its
confirmation gate. Resampling changes probabilities but didn't improve ranking.

**17. Hyperparameter tuning?**
LR C = 0.1 (L2). A 120-trial Optuna search on an LR + XGBoost blend: 77/120 trials within
0.002 of the best, best fresh-split gain +0.0017, under our pre-set bar, so not adopted.
The surface is flat; the data, not the knobs, limits performance.

**18. Are the scores calibrated probabilities?**
Broadly: mean OOF score 0.0176 vs 1.77% fraud rate, and rank-band observed rates fell
inside their intervals in our error audit. Don't claim point-wise calibration; say "scores".

## E. Validation and leaderboard

**19. How do you know you're not overfitting?** (slide 9)
Stratified 5-fold, everything refit in-fold, 5 repeats × 5 folds (0.2330 mean, SD 0.0046),
adoption rules written before results, failures recorded. *Caveat:* the same 353 labels
were reused across many experiments, so some selection optimism remains.

**20. How noisy is your CV?**
Per-fold PR-AUC on the reported split: 0.160–0.326 (~70 frauds per fold). An earlier
blend-vs-LR bootstrap gave +0.0028 with 95% interval [−0.0038, +0.0096]: individual
small gains are within noise, which is why we needed repeats and confirmations.

**21. Why is public (0.189) lower than CV (0.24)?** (slide 12)
The test set is shifted: train-vs-test classifier AUC 0.67; gaps 4.9% → 13.9%, new devices
9.1% → 15.4%, 90th-pct amount $530 → $1,707. We tested shift reweighting (#2541, 0.18721)
and device-segment label-shift correction (even an oracle gains ≤ 0.008); neither explains
it. Honestly unresolved.

**22. Did you pick models from the leaderboard?**
No. Public slice ≈ 60–100 frauds: one score swings ±0.04; two related blends differ by
±0.004–0.006. Rank correlation of local CV vs public across 11 submissions: −0.06
(#964 had the best CV and the worst public). #2756, #2510 and #1486 were statistically
tied on CV; #2756 broke the tie with the best public score and a fully verified package.

## F. Operations and business

**23. How would you pick the threshold?** (slide 10)
From review capacity. Best-F1 cut-off 0.194: 83 caught / 105 false alarms, precision 44%,
recall 23.5%. Cut-off 0.05: 137 caught / 788 false alarms. On test, 0.194 flags 1.55% vs
0.94% in validation, so re-check the cut-off live.

**24. Block the card or not?**
No: the score triggers step-up verification (OTP / 3-D Secure). A false alarm costs seconds,
not a blocked card. Top 1% of traffic is 42% fraud; top 0.5% is 61%.

**25. How would you deploy and monitor it?**
Single pickled pipeline + notebook, CPU only, one extra dependency (catboost 1.2.10).
Monitor score distribution and alert rate, missingness rate, and confirmed-fraud precision;
retrain and re-pick the cut-off on drift. Labels arrive late and only for investigated
alerts, so monitoring must allow for that.

**26. Fairness: does the country flag penalise foreigners?**
Non-SG countries do run higher (1.5–4.1% vs SG 1.31%), but it's an association. Mitigations:
step-up rather than block, monitor alert and false-alarm rates by country, and test the
model with and without the flag if deployed.

## G. Errors and limits

**27. What does the model miss?** (slide 11)
194 of 212 frauds ranked below 1,000 are on known devices, with ordinary amounts and
activity. PR-AUC within known-device rows 0.049 vs 0.512 for new devices. They look like
legitimate traffic; per-customer history is needed.

**28. Did you look for leakage or a "magic feature"?** (slide 12)
Yes, and we closed it: hidden account IDs from shared account_age (P(fraud | group-mate is
fraud) only 1.9–4.7%), transaction-chain rebuilds (1 chain vs 1 in a shuffled control),
eight value fingerprints (all explained by the score, CMH p ≥ 0.49).

**29. What would you do with more data/time?** (slide 13)
Customer ID + timestamps for rolling per-card behaviour features (Wedge et al. 2017 cut
false positives 54%), device fingerprinting, drift monitoring.

**30. Is ~0.24 the ceiling?**
Don't claim one. Say: none of our ideas beat ~0.24 CV; that's not proof none can.

## H. Reproducibility

**31. How do you know your submission runs?** (slide 3)
Re-ran the uploaded notebook + model.pkl in a clean kernel on shuffled, ID-free test input
on 4 Oct: 12,000 rows, max difference 0.0 vs the scored CSV. Earlier Linux replay matched
within 2.2e-16. Unseen categories, single rows and batch size all give valid output.

---

## Don't say
- That the public score predicts the final ranking.
- That any feature *causes* fraud, or "new device = account takeover".
- "Risk factors combine multiplicatively" without the corner exception (slide 5).
- "9–34%" for the new-device × young-account corner (recomputed: 18.4%; 11–61% by amount band).
- A performance ceiling.
