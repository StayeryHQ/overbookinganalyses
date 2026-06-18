# Open decisions

Methodological choices that the notebooks now expose but haven't *finally*
decided. Each one is implemented with a default — change the default if
you disagree.

| # | Decision | Current default | Alternative | Where to change | Recommended when... |
|---|---|---|---|---|---|
| 1 | Static-model granularity for rate plan | `ratePlan_category` (59 mid-grain buckets) | `ratePlan_type` (6 coarse buckets) | Cell 6 (feature roster), 01-03 | …you want presentation-friendly interpretability over raw predictive power. Trees won't care; LogReg might. |
| 2 | High-cardinality country encoding | **DECIDED**: `guest_country_region` (DACH/GB/EU_other/RoW/Unknown), built once in 00 via `src/features.py` | target encoding via `category_encoders.TargetEncoder` (pipeline-only, target-dependent) | `src/features.py` (taxonomy); built in 00 §3.0.i | …a rare-country slice shows the region split is too coarse. Target encoding tied region in the benchmark but is target-dependent, so it must stay a fitted pipeline step (never in 00). |
| 3 | Threshold strategy | **UPDATED 2026-06-11**: argmax F1 on the dedicated **val block** (see Resolved) | argmax (precision × revenue) — i.e. weight false negatives by `gross_amount` | §8, 01-03 | …you decide it's better to flag fewer-but-bigger-revenue bookings. The OOF approach is methodologically clean; the revenue-weighted threshold is operationally honest. |
| 4 | RF vs HistGradientBoosting | **DECIDED 2026-06-11**: HistGB; RF is out of the lineup (see Resolved) | - | notebook 03 (HistGB, to build) | …RF underperforms XGBoost by > 0.01 AUC. HistGB is faster, better calibrated, handles NaN natively. |
| 5 | LogReg regularisation | **TUNED 2026-06-16 in §6b** (RandomizedSearch over `C` + `l1_ratio`, TimeSeriesSplit, AP); ElasticNet `l1_ratio=0.5` is the baseline fallback | Pure L1 (`l1_ratio=1.0`) or pure L2 (`l1_ratio=0.0`) | §5 / §6b, notebook 01 | …done; the search covers the full L1↔L2 range. |
| 6 | XGBoost hyperparams | **IMPLEMENTED 2026-06-16 in 02**: early stopping (n_estimators) + RandomizedSearchCV (40 iters, TimeSeriesSplit, scored on AP) on train; baseline-vs-tuned compared on val, winner kept | manual `n_estimators=600, max_depth=6, lr=0.05` as fallback baseline | §5 / §6c, notebook **02** (not 03 — that's HistGB) | …done; widen the search space only if §6c shows a clear gradient. |
| 7 | Calibration method | `isotonic` | `sigmoid` (Platt scaling) | §5, 01-03 | …test set is small (< ~1000 rows). Isotonic needs ~1000+ rows to fit reliably; sigmoid is more robust on small N. We have plenty of data so isotonic is fine. |
| 8 | Temporal-test share | **DECIDED**: 60/15/25 train/val/test by arrival (00 §7) | 10% / 30% / a fixed cutoff date | Cell 95, notebook 00 | …you want a longer training window (smaller test share) or a longer hold-out (larger). 25% is the textbook default; consider 10% if you start retraining weekly. |
| 9 | TimeSeriesSplit folds | 5 | 3 (faster) or 10 (lower variance estimate) | §6, 01-03 | …CV runtime becomes a problem (10×) or you want tighter variance on the headline number (10 folds). |
| 10 | Hazard snapshot days | `{1, 3, 7, 14, 30, 60, 90}` | finer grid `{1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90}` | Cell 4, notebook 08 | …per-snapshot AUC variance suggests we're undersampling the curve in the 1-7d region (the operational sweet spot). Doubles row count. |
| 11 | Hazard target window | "cancel between snapshot d and previous snapshot" | "cancel within next k days" (overlapping windows) | Cell 4, notebook 08 | …the gappy snapshot scheme creates label noise. Overlapping windows are cleaner but trickier to evaluate. |
| 12 | Cancel-timestamp source | **DECIDED**: true `cancellationTime` (raw), else `modified`, else midpoint | true `cancellationDate` (if BigQuery exposes it) or uniform random within lead window | Cell 4, notebook 08 | …cancellationTime coverage (printed in §1) is low, so the `modified` fallback dominates — otherwise the hazard model is learning a noisy target. |
| 13 | XAI sample size for PDP/SHAP | 1000 PDP / 500 SHAP | 5000 PDP / 2000 SHAP | §11, 01-03 | …notebook is going into a presentation deck (more rows = smoother curves). Cost: ~5× slower XAI. |
| 14 | Ensemble strategy in 05 | mean of probabilities | stacked LogReg on top of (`p_logreg, p_xgb, p_histgb`) using OOF predictions (lineup 2026-06-11; RF/MLP are out) | Cell 18, notebook 05 | …mean ensemble lifts AUC by < 0.005. Stacking will help if the four models have meaningfully different error patterns. |
| 15 | Risk thresholds for buckets | `low < 60%`, `60-75% uncertain`, `≥ 75% high` | Calibrated by historical realised cancel rate (e.g. set `high` so the realised rate in the high bucket is ≥ 85%) | `src/scoring.py` `LOW_THR` / `HIGH_THR` | …after a month of post-calibration scoring, recompute the realised rate per bucket and shift the thresholds to keep operational meaning. |

---

## ⚠️ FLAGGED 2026-06-12 - guest-profile outcome leakage (decision pending)

`primaryGuest_address_countryCode` (-> `guest_country_region`),
`primaryGuest_preferredLanguage` and (milder) `travelPurpose` are completed at/around
check-in. Cancelled bookings keep empty profiles; future Confirmed bookings - the
actual scoring population - are 38% empty on country. Training data therefore encodes
the outcome in the missingness ("Unknown" = 90.6% cancel rate in train+val).

Implications: all offline AUCs inflated; country-encoding benchmark partially
invalidated (its "drop" arm lost the leak, not just signal). Options on the table:
(a) quantify first - re-run 01 with vs without the three fields;
(b) remove the three fields from all model rosters until booking-time snapshots exist;
(c) start snapshotting these fields at refresh time so clean booking-time values
    accumulate going forward.
Discovered by experiments/cancel_rate_by_feature.ipynb on its first run.

**QUANTIFIED 2026-06-12** (experiments/profile_leakage_quantification.ipynb,
train->val, test untouched): offline inflation +0.098 AUC (HistGB 0.9172->0.8197)
resp. +0.117 (LogReg 0.8828->0.7655); leaky models predict 82-88% cancel prob on
profile-missing rows (= the state of ~38% of the scoring population); real signal on
filled-only rows +0.056-0.062 AUC. Recommendation: remove `guest_country_region`,
`primaryGuest_preferredLanguage`, `travelPurpose` from all rosters now; start
booking-time snapshotting in the refresh flow to re-admit clean versions later.
Option (a) done; **(b) DONE 2026-06-12** - fields removed from all model rosters (src/features.py NON_FEATURE_COLS + src/scoring.py lists + 01/benchmarks); columns stay in the clean parquet for diagnostics. (c) booking-time snapshotting still open.

## Resolved (with evidence)

### Leakage hardening (split-first for the 2 fitted steps) + 01 tuned (2026-06-16)

User flagged the classic "engineer-then-split" leakage worry. Audit of 00 showed the
**vast majority of features are row-wise** (log, ratios, date parts, flags, region map)
→ leak-free regardless of split order, and the high-risk fitted preprocessing
(impute/scale/OHE) is already train-only in the model pipelines. Company history (§5.2)
verified leak-safe: sorted by `created`, `cumcount`/`cumsum` exclude the current row →
strictly-earlier only. The group-mode imputation (§5.1) only fills `preferredLanguage`
(a removed leakage field) → moot.

Two genuine full-frame-fitted steps were **hardened to train-only** ("gezielt härten",
fit-on-train / transform-all; train = oldest 60% by arrival, matching §7):
- **§3.0.d** `ratePlan_category` rare-category folding — `value_counts` now on train rows.
- **§4.3 Rule 5** `gross_per_night` MAD cap — median+MAD now on train rows.
Other outlier rules (lead>365, los<1, gross≤0, los>200) are fixed domain thresholds →
no leak, untouched. Measured impact of the change is ~0 (per the earlier folding study),
but the audit trail is now clean. Requires a 00 re-run to refresh the parquet.
Note: the train mask is computed locally at each step (≈ final split); for bit-exact
alignment one would freeze the split date early — deferred as the larger "voll refactoren"
option.

**01 (LogReg) tuned + brought to 02's level (resolves #5 / the §13 "ungetuned" point):**
new **§6b** — RandomizedSearch over `C` + `l1_ratio` (TimeSeriesSplit on train, scored on
AP) + baseline-vs-tuned on val, winner kept; provenance in the model card +
`baseline_vs_tuned_val.csv`. 01 ran clean end-to-end before the change (the only mentions
of removed fields were in explanatory markdown, not code).

### Notebook 02 (XGBoost) built + numbering fixed (2026-06-16)

`notebooks/02_xgboost.ipynb` authored as a faithful sibling of 01 (identical
features/splits → fair comparison in 05) and **smoke-validated end-to-end** on a
reduced `FAST_DEV` sample (all 13 sections run, all artefacts written). Awaiting a
full run for headline numbers. Numbering clarified project-wide: **02 = XGBoost,
03 = HistGB** (decision #6 previously mislabelled XGBoost as "notebook 03").

Methodology choices baked into 02 (all transparent + auditable in the notebook):
- **Encoding decided empirically** (§6b mini-benchmark, onehot vs native-categorical
  vs ordinal on TimeSeriesSplit): tied at this low cardinality (max ~35 levels), so
  **one-hot** is kept — SHAP-robust, 100% built-in (loads in app/scorer with no extra
  class), directly comparable to the LogReg baseline.
- **No scaling** (trees are scale-invariant).
- **`company_prior_cancel_rate` keeps NaN** (passthrough → XGBoost native missing
  handling) instead of 01's forced 0-fill, so "no company history" is no longer
  conflated with "0% historical cancel rate". Methodological gain over 01.
- **XAI**: native gain importances (collapsed to parent features) + permutation +
  PDP/ICE + **exact TreeSHAP via XGBoost `pred_contribs`** (version-robust; SHAP's
  own TreeExplainer mis-parses xgboost-3.2 model dumps).

`src/scoring.py` MODEL_REGISTRY updated to the real lineup (logreg/xgboost/histgb;
rf/mlp removed) and the `best_model_by_auc()` bug fixed (it read `test_metrics["roc_auc"]`
but the cards store `["auc"]` → it would have raised KeyError). `main.py --model`
choices updated to match. **Still open:** `src/scoring.py build_features` does not yet
compute the company-history bundle, so the production scorer can't serve the 24-feature
models end-to-end — pre-existing train/serve parity TODO, independent of 02.


### Folding thresholds + free-cancel features + benchmark metrics (2026-06-11)

**Folding DECIDED (frequency_folding experiment, re-run on TRUE raw categories):**
`channelCode` keeps all 18 raw levels (no folding, k=1); `ratePlan_category`
min_count raised 20 -> 50. Hold-out AUC was flat across the whole sweep (HistGB
0.9219-0.9228, LogReg 0.8788 throughout) and the train-vs-full-frame leakage delta
was exactly 0.0 at the base thresholds. Implemented in 00 §3.0.a / §3.0.d.
Requires a 00 re-run to refresh the clean parquet.

**Free-cancel features REMOVED:** `free_cancel_days_before_arrival` +
`has_free_cancel_window` dropped from 00/01/benchmark rosters. Reasons: the flag was
constant 1 (dueDateTime set on 100% of rows); for nonref the value duplicated
lead_time_days exactly (corr 1.00); source data judged unreliable - signal lives in
ratePlan_type/category + lead_time anyway. Audit stub kept at 00 §3.0.f2.

**Survival benchmark metrics DECIDED:** headline = per-snapshot AUC@d + Brier@d +
expected-vs-realized cancellations on d in {90,60,30,14,7,3,1}; C-index demoted to a
sanity footnote (it answers "who cancels sooner", not the overbooking question).
Implemented in experiments/survival_benchmark.ipynb; smoke run already shows the
static classifier over-forecasting ~2.9x at d=1 while the discrete hazard stays ~1.0.

**New 00 §5.3:** association re-check (Spearman + Cramér's V) for the late-built
company-history bundle, so §3.6/§3.7-style audits also cover §5.2 features.
First run flagged: Cramér's V has_company vs travelPurpose = 0.65 (companies book
Business) - known overlap, kept; both features stay.


### Lineup + split + threshold strategy (2026-06-11)

**Model lineup DECIDED:** static = LogReg, XGBoost, HistGradientBoosting;
survival = discrete-time hazard (08), XGBoost AFT, Random Survival Forest.
RandomForest and MLP are **out** (resolves #4 in favour of HistGB; archive
README 2026-06-09 already pointed this way). Evidence: survival_benchmark —
static HistGB best cancel-AUC (0.92), XGB AFT best C-index (0.87), RSF a solid
all-rounder; no single model dominates both metrics, hence two layers.

**Threshold tuning DECIDED (supersedes #3 default):** tuned on the dedicated
**val block** of the 60/15/25 temporal split from 00 §7 (not on OOF train
predictions). Test stays untouched until the final headline run.
Revenue-weighted thresholds remain the open alternative in #3.


### #2 — country encoding → `guest_country_region` (2026-06-08)

Replaced the per-notebook collapsers with a single structural region feature
built in 00 (`src/features.py`, used by 00, 01–04, 08 and `src/scoring.py`).

Why: the old setup was both inconsistent and sub-optimal. 01 used
`RareCategoryCollapser(min_count=10)` while 02–04/08 used
`TopKCategoryCollapser(k=5)` — different feature spaces for the same variable,
silently biasing the 05 comparison. A controlled benchmark on the **temporal
hold-out** (`notebooks/experiments/country_encoding_benchmark.ipynb`,
`reports/tables/00_audit/country_encoding_benchmark.csv`) tested six encodings
for two model families. HistGradientBoosting results:

| encoding | AUC | AP | rare-slice AUC |
|---|---|---|---|
| **region** | **0.907** | **0.759** | **0.964** |
| target_oof | 0.906 | 0.758 | 0.963 |
| mincount10 | 0.905 | 0.754 | 0.958 |
| topk5 (old trees) | 0.884 | 0.677 | 0.886 |
| is_international | 0.879 | 0.662 | 0.886 |
| drop | 0.817 | 0.552 | 0.795 |

`region` tied the best (within bootstrap CI of target_oof and mincount10) while
being structural (leakage-free → can live in 00), the leanest multi-level option
(7 cols), scalable to new markets, and interpretable. `topk5` left ~0.08 AP on
the table. Note: rare-country bookings cancel at 0.439 vs 0.198 overall, so this
is a top-tier feature — investigate *why* (proxy for non-refundable rate plans /
OTA channel?) during XAI.

### Arrival window floor - exclude pre-2022-08 (COVID) regime (2026-06-08)

`00 2.6` now drops arrivals before `2022-08-01`. The COVID era (2020-2021) shows
cancel-rate spikes to ~0.59 and an Omicron-winter bump into early 2022; the
series only settles to a stable ~0.20-0.25 from mid-2022. Training on the old
regime teaches patterns that no longer hold. Evidence:
`experiments/cancellation_rate_over_time.ipynb`. The floor + dynamic future
cutoff + run timestamp are stamped in `Data/reservations_clean_meta.json`.

### Company encoding -> engineered history bundle (2026-06-09)

`company_name_clean` (11,660 clustered companies) now feeds leakage-safe history
features in 00 §5.2: `has_company`, `is_repeat_company`, `company_prior_bookings`,
`company_prior_cancel_rate` (counts over strictly-earlier bookings only). The
`history` arm was the best company representation on the temporal hold-out
(HistGB AUC 0.9124 / AP 0.7748 vs `drop` 0.9102 / 0.7698; prior-cancel-rate
quintiles 0.086 -> 0.315). Naive `target_oof` was *worse* than dropping company.
Evidence: `reports/tables/00_audit/company_encoding_benchmark.csv`.
Modelling-phase TODO: add to 01-05 rosters + replicate counts in `src/scoring.py`
(train/serve parity) before deployment.

### CatBoost -> not added (2026-06-09)

CatBoost's edge is automatic target-encoding of high-card categoricals; the
experiments show that mechanism does not help here (`target_oof` lost to `region`
for country and to `drop` for company). `region` (structural) + company history
cover the categoricals. No CatBoost dependency; keep OOF target encoding only
where it demonstrably wins (none so far).

### Frequency folding -> keep 00 thresholds; full-frame folding is safe (2026-06-09)

Sweeping `min_count` for `channelCode` / `ratePlan_category` left hold-out AUC flat
(~0.9108); fitting the keep-set on the full frame vs train-only moved AUC by
0.000 / -0.0003. 00's `<100` / `<20` full-frame folding is safe and the exact
threshold is not load-bearing. Evidence:
`reports/tables/00_audit/frequency_folding_experiment.csv`.

### Validation split -> 60/15/25 train/val/test (2026-06-09)

00 §7 creates `temporal_split` (train oldest 60% / val 15% / test newest 25%) plus
`is_temporal_test` / `is_temporal_val`. Tune threshold + hyperparameters on **val**,
touch test once; deploy by retraining on 100% with the operating point **locked
from val**. Fixes the old bug (F1 threshold tuned on test). Use a *transferable*
operating point (top-X% risk / precision >= Y), not an absolute probability, since
calibration drifts across retrains. Modelling-phase TODO: wire 01-05 to consume
`temporal_split`.

### Model lineup -> LogReg + RF + XGBoost + HistGB; drop MLP (2026-06-09)

HistGradientBoosting added (fast, calibrated, strong in the experiments); MLP
dropped (weak tabular baseline). Plus the discrete-time hazard (08). CatBoost
excluded.

### Layering -> static + hazard, plus a stacked ensemble to try (2026-06-09)

Layer by time-horizon: static model for long-lead bookings, discrete-time hazard
(08) for the near-arrival daily-rescoring view. Additionally try a stacked
meta-learner (LogReg on the base models' OOF probabilities) in 05; keep it only if
it beats the best single model beyond noise.

### Survival analysis -> benchmark before adopting (2026-06-09)

`experiments/survival_benchmark.ipynb`: static classifier vs XGBoost AFT vs Random
Survival Forest + Gradient-Boosted Cox (scikit-survival), scored by concordance on
the temporal hold-out. DeepSurv excluded (overkill at this scale, PH assumption,
hard to audit). Adopt a survival model only if it clearly beats the static
C-index; otherwise the discrete-time hazard (08) stays the survival tool. Requires
`pip install scikit-survival`.

## Two larger decisions worth flagging separately

### A. Property column granularity

We use `property_name` (11 unique). The original notebooks used
`property_code` which is dropped in 00 §3.3. Both are equivalent for
modelling — pick whichever reads better in dashboards.

### B. Time-of-week vs. month for arrival features

We have `arrival_dow` (day of week) and `arrival_month`. The dow signal
will be strong (weekend leisure vs midweek business); the month signal
will pick up seasonality but blurs across week/weekend within a month.

Worth checking via the SHAP scatter plot in 09: if `arrival_dow` lights up
strongly but `arrival_month` is flat, your bookings don't have meaningful
month-level seasonality and `arrival_month` is dead weight (you could
remove it for cleanliness).
