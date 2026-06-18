> ⚠️ **PARTIALLY SUPERSEDED (2026-06-12).** This review reflects the 2026-06-09 state.
> Since then: RF + MLP are OUT of the lineup (LogReg/XGBoost/HistGB + hazard/XGB-AFT/RSF);
> threshold tuning moved to the dedicated val block (60/15/25); channel folding removed;
> ratePlan min_count = 50; free-cancel features removed; guest-profile leakage discovered
> and quantified (+0.10-0.12 AUC inflation). Sections discussing RF (02) and MLP (04) are
> historical. Current truth lives in reports/open_decisions.md.

# Model pipeline review - sequenced roadmap

Living review, ordered by the sequence we actually work through (not by severity).
Tags: **[DONE]** resolved · **[NOW]** active this phase · **[NEXT]** immediately
after · **[LATER]** deliberately deferred - do not action yet.

We are in **Phase 0 (notebook 00)** + **Phase 1 (model selection & layering)**.
Evaluation rigor (bootstrap CIs, per-slice metrics) and hyperparameter search are
**Phase 3-4 - deferred on purpose**.

---

## Phase 0 - Data audit (notebook 00)

- [DONE] Target & population: positive = Canceled, negative = CheckedOut. NoShow
  excluded (room blocked shortly after check-in, cannot be resold). InHouse kept as a negative (checked in -> did not cancel before arrival; resolved). Confirmed dropped (unresolved).
- [DONE] Arrival window: floor 2022-08-01 (exclude COVID regime) + dynamic future
  cutoff today-3d (label completeness / recent-window selection bias). Run metadata
  stamped to `Data/reservations_clean_meta.json`.
- [DONE] Leakage: `cancellationTime` + `diff_cancellation_arrival` dropped in §3.2
  (08 derives cancel timing from raw); `is_first_res`/`is_last_res`, balance and
  payment-account flags, `modified` dropped.
- [DONE] Country -> `guest_country_region` (structural, `src/features.py`).
- [DONE] Company -> history bundle (§5.2); CatBoost not needed.
- [DONE] Frequency folding verified safe (full-frame; thresholds non-load-bearing).
- [DONE] 3-way temporal split 60/15/25 (`temporal_split`).
- [NEXT] §3.6.5 numeric<->categorical (eta): confirm `diff_gross_cancellation_fee`
  redundancy with `ratePlan_type` once 00 is re-run.
- [LATER] Remaining §3.1-3.7 walkthrough (trios, near-constants, multicollinearity).

## Phase 1 - Model selection & layering (NOW)

- [SUPERSEDED 2026-06-11 - RF also out now] Lineup was: LogReg + RF + XGBoost + HistGB + discrete-time hazard (08); MLP
  dropped, CatBoost excluded.
- [DONE] Layering: static (long-lead) + hazard (near-arrival); stacked ensemble
  (LogReg on OOF probs) to TRY in 05, kept only if it beats the best single model.
- [NOW] Survival check: `experiments/survival_benchmark.ipynb` - adopt only if it
  clearly beats the static C-index.

## Phase 2 - Honest evaluation wiring (NEXT)

- [NEXT] Wire 01-05 to `temporal_split` (the notebooks currently ignore the temporal
  hold-out and use a random split - bias A1/B1 below).
- [NEXT] Tune the operating threshold on VAL, not test (fixes A5). Transferable
  operating point (top-X% / precision>=Y), not an absolute probability.
- [NEXT] Calibration: wrap final classifiers in `CalibratedClassifierCV` OR drop
  class-weighting - not both (A4). `p_cancel` feeds revenue-at-risk -> load-bearing.
- [NEXT] Company history -> rosters + `src/scoring.py` (train/serve parity).

## Phase 3 - Evaluation rigor (LATER)

- [LATER] `TimeSeriesSplit` as primary CV (B1).
- [LATER] Bootstrap CIs on headline metrics (B2).
- [LATER] Precision@k / lift-at-decile + per-slice AUC by channel / ratePlan /
  lead-time bucket (B3/B4).

## Phase 4 - Tuning & productionization (LATER)

- [LATER] `RandomizedSearchCV` per model; pin params in `configs/` (C1).
- [LATER] Hazard: finer snapshot grid, target window, true `cancellationTime` (now
  wired in 08), per-lead-time evaluation (D).

---

# Appendix - original detailed findings (severity-ordered, preserved for traceability)

# Critical review — notebooks 01–05

Senior-DS pass on the modelling pipeline as it stands today. Items are ordered by
severity: bugs that fail silently first, then issues that bias evaluation, then
design choices that are defensible but worth questioning.

---

## A. Silent bugs / things that don't do what the code claims

### A1. The temporal hold-out exists but is ignored
`00 §7` adds an `is_temporal_test` column flagging the newest 25% of rows by
arrival. The model notebooks then **do not use it** — they do a plain
`train_test_split(stratify=y)` with `random_state=42` instead.

Why this is a problem: cancellation behaviour is non-stationary. Channel mix
shifts (Booking.com vs Direct vs corporate), pricing changes, policy changes
(non-refundable rates, COVID, etc.) all move the joint distribution. A random
split tests how well you predict the past from the past. A temporal hold-out
tests whether you can predict next week from last week — which is the only
thing that matters for daily scoring (`06`).

If a temporal AUC is materially worse than a random-split AUC, the deployed
model is over-promising. Right now you cannot tell, because the temporal
hold-out is dead code.

**Recommendation:** in cells 8 of 01–04, split as:
```python
test_mask = df["is_temporal_test"].astype(bool)
X_train, X_test = X[~test_mask], X[test_mask]
y_train, y_test = y[~test_mask], y[test_mask]
```
Keep the random-split numbers in the model card as a secondary check, but
**lead with the temporal hold-out** in `05_model_comparison` and downstream.

### A2. `property_code` references after the column is dropped (FIXED)
Per-property AUC in cell 21 of every model notebook grouped by `property_code`,
but that column is dropped in 00 §3.3. The chart was failing silently with an
empty groupby. I patched all four notebooks to use `property_name` (11 unique
properties; the surviving column). Verify by re-running cell 21.

### A3. Engineered features referenced but never built
`arrival_dow`, `arrival_month`, `is_weekend_arrival`, `stay_bucket` are listed
in `00 ENGINEERED` and `COLUMN_DESCRIPTIONS`, used in every model notebook,
and **never built in any code cell** before today. I added the builder cell as
`§3.0.h`. Re-run notebook 00 once before running 01–04 so the parquet contains
them.

### A4. `class_weight='balanced'` / `scale_pos_weight` distorts probabilities
- `01 LogReg` uses `class_weight="balanced"`.
- `02 RF` uses `class_weight="balanced_subsample"`.
- `03 XGB` uses `scale_pos_weight = neg/pos`.
- `04 MLP` uses no class balancing.

Each of these **down-weights the majority class during training**, which is
fine for ranking metrics (AUC, AP) but pushes the predicted probability scale
upward. Your test-set Brier scores will look worse than they should, and any
threshold tuned at p=0.5 is no longer interpretable as "the model thinks this
booking is more likely than not to cancel".

You then plot reliability diagrams in 05 §5 expecting calibration, but the
training-time reweighting has made them inherently miscalibrated. **Either**
drop the class-weighting and let the natural prior come through (then use
`average_precision` to compare), **or** keep the class-weighting and wrap each
model in `CalibratedClassifierCV(method="isotonic", cv=5)` before saving
predictions. Mixing the two as you do now is the worst of both worlds.

### A5. Threshold tuning is done on the held-out test set
Cell 17 picks the F1-optimal threshold by argmaxing F1 on `(y_test, y_prob_test)`.
That's the same set used to report headline metrics in cell 15. Picking the
threshold on the evaluation set leaks the evaluation set into the operating
point — your reported precision/recall at "best F1" are upward-biased.

**Recommendation:** carve a 20% validation slice off the training set (or use
CV out-of-fold predictions) to pick the threshold, then apply it once to the
test set. Easiest fix: `oof = cross_val_predict(pipeline, X_train, y_train,
cv=cv, method="predict_proba")[:, 1]` and tune on those, since you're already
running 5-fold CV.

---

## B. Evaluation choices that flatter the numbers

### B1. 5-fold stratified CV is the wrong CV for time-series data
`StratifiedKFold(shuffle=True)` randomly mixes recent and old rows into the
same fold. The model can learn from a row arriving in 2024-Q3 to predict a
row arriving in 2024-Q2 — which is information you never have at scoring time.
This is the same bias as A1 but inside CV.

**Recommendation:** for the headline number, use
`sklearn.model_selection.TimeSeriesSplit(n_splits=5)` over rows sorted by
`arrival`, or do a single chronological train/val/test split (e.g. train ≤
arrival_q60, val arrival_q60..arrival_q80, test = `is_temporal_test`). Keep
the stratified random CV as a secondary diagnostic for stability across the
imbalanced classes.

### B2. The same `random_state=42` everywhere disguises real disagreement
All four models share the exact same train/test rows, which is good for fair
comparison in 05. But because the F1 threshold is tuned on the test set
(see A5), a small change in seed will change which model "wins" on F1 — and
you have no way to detect that. Report each model's metric with a bootstrap
confidence interval (`scikit-learn-extra` or hand-rolled
`np.random.choice(..., replace=True)`), not a single point estimate.

### B3. AUC headline is dominated by easy negatives
Class share is ~21% positive (post §4.5). AUC is fine as a ranking metric, but
your business cost is asymmetric: failing to flag a cancellation loses a
night's revenue; flagging a non-cancellation costs ~nothing (you don't actually
oversell against it, you just nudge the property). Lift at top-k% (or
precision at fixed recall) is a more honest metric. Add a "precision@10%
volume" / "precision@50% recall" column to the headline table in 05 §2.

### B4. Per-property AUC heatmap omits the *interesting* slices
Per-property is one cut. The cuts that would actually expose model failure
modes are:
- per channel (`channelCode`) — does the model work for OTA but not direct?
- per `ratePlan_type` (nonref vs flexible vs corporate) — non-refundable
  bookings cancel very differently and you have a strong prior here.
- per lead-time bucket — bookings made <7 days vs >60 days have entirely
  different cancel hazards.
- per stay-bucket — short / mid / long extend-stay segments behave differently.
A per-property heatmap is a vanity chart by comparison.

---

## C. Modelling choices that are defensible but I'd push back on

### C1. No hyperparameter search anywhere
- RF: `n_estimators=400, max_depth=None, min_samples_leaf=20, max_features="sqrt"` — fine defaults, but no search.
- XGB: `n_estimators=600, max_depth=6, lr=0.05, subsample=0.8` — reasonable, but on 150k rows you should be doing a small `RandomizedSearchCV` (n_iter=30) over `(max_depth, learning_rate, min_child_weight, reg_lambda)`.
- MLP: `(64, 32)` ReLU, `alpha=1e-3` — extremely under-explored.

For a cancellation model going into production daily scoring, at least one
randomised search per model is table stakes. Pin the best hyperparams in a
`configs/<model>.yaml` so the run is reproducible.

### C2. OHE for `primaryGuest_address_countryCode` (182 unique) is wasteful
That's 182 sparse columns, most of which appear in <20 rows. For tree models
prefer **target/CatBoost-style encoding** or **top-k frequency encoding**
(keep DE/NL/GB/AT/CH explicitly, fold everything else into "Other"). For the
linear model, OHE with `handle_unknown="ignore"` is OK but you're paying for
features that contribute almost nothing.

### C3. Logistic regression with L2 only — try L1 / ElasticNet
With ~80 OHE columns you'd benefit from L1 (or ElasticNet) to drive
coefficients of useless dummies to zero, both for interpretability and for the
"interpretable coefficient" claim in the notebook header. Cost: switch
`solver="liblinear"` → `solver="saga"`, add `l1_ratio`.

### C4. Random Forest with `max_depth=None` + 150k rows
Trees grow until leaves are pure or hit `min_samples_leaf=20`. With 150k rows
of training data and ~80 features, individual trees can be enormous. That's
fine for AUC but the model object is large, slow to load in `06_daily_scoring`,
and tends to be wildly overconfident (probabilities cluster at 0 and 1).
Try `max_depth=12-16`, or better: replace RF with HistGradientBoosting (faster,
calibrated by default, handles missing values without imputation).

### C5. The MLP architecture is uninformative for tabular
`(64, 32)` ReLU with no batch norm, no dropout, no embeddings for high-card
categoricals. Tabular MLPs at this scale need either (a) entity embeddings
for categoricals and a wider/deeper net, or (b) acknowledge they're a sanity
floor against the tree models. The current MLP is the latter — fine, but
don't expect it to win.

### C6. No probability calibration step
Trees and boosting models output probabilities that are systematically
miscalibrated (XGBoost especially). If anyone downstream uses the
probability as a continuous risk score (e.g. expected-revenue-at-risk =
`p_cancel * gross_amount`), the math is wrong. Wrap the final classifier in
`CalibratedClassifierCV(method="isotonic", cv=5)` before `pipeline.fit` and
your reliability diagrams will look like reliability diagrams.

### C7. Notebook 05's "simple ensemble (mean of probabilities)" is wishful
Simple averaging only helps when the models' errors are uncorrelated. RF /
XGB / MLP all train on the same OHE'd matrix and disagree mostly on tree depth
vs neural nonlinearity — high error correlation. The lift from averaging is
usually 0.001–0.005 AUC and that's within noise. If you want a real ensemble:
stack with a meta-learner (LogReg on the four `p_cancel` outputs + 3–4 raw
features), trained on out-of-fold predictions only.

---

## D. The big architectural question I'd raise in design review

**You're treating this as a static-feature binary classification problem.
Cancellation is actually a survival / hazard problem.**

A booking made 90 days out has a different cancellation risk profile at day 89
than at day 30 than at day 2. The current model conflates all of them into a
single "P(cancel ever)" trained at booking creation time. That's why
`time_till_arrival_today` and `time_since_booking_today` appear in the data
dictionary — they're dynamic features that move every day.

If you want the daily-scoring (`06`) numbers to be honest, you have two
options:

1. **Per-day rescoring with the static model** (what you're doing now). Acceptable, but the probability drifts only because the *features* drift, not because the model captures hazard. You'll be over-confident on long-lead bookings and under-confident as arrival approaches.
2. **Survival model** (Cox / discrete-time hazard / a per-day LSTM). Substantially more work; pays off when revenue-at-risk calculations need to integrate over time-to-arrival.

For a v1 production model, (1) is fine — but the notebooks should explicitly
state that's the choice, and the comparison should split test AUC by lead-time
bucket to expose where the static model is weakest.

---

## TL;DR — what to do this week

Tier 1 (do before retraining):
1. Wire 01–04 to the temporal hold-out (`is_temporal_test`).
2. Move F1-threshold tuning off the test set (use OOF CV predictions).
3. Wrap each classifier in `CalibratedClassifierCV` *or* drop class-weighting — pick one.

Tier 2 (do before deployment):
4. Add `TimeSeriesSplit` as the primary CV.
5. Add precision@k and lift-at-decile to the headline table.
6. Slice 05's per-property heatmap by `channelCode`, `ratePlan_type`, and lead-time bucket.

Tier 3 (when you have time):
7. RandomizedSearchCV per model, pin best params in `configs/`.
8. Replace OHE for `countryCode` with top-k or target encoding.
9. Decide explicitly whether v1 stays as static classification or moves to
   discrete-time hazard.
