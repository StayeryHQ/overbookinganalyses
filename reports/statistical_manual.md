# STAYERY Overbooking - Statistical & Methods Manual (v11)

> Deep reference. Every transformation, model, parameter, metric, and decision rule that the
> shipped pipeline uses, with the math and the *why*. The companion `pipeline_reference.md` is
> the short mental map; this file is the place to become expert on the details.
>
> Scope note: this describes the **current** pipeline (target = cancel-before-arrival, `_log`
> features, cost-based threshold, models 01/02/03 + hazard 08). RandomForest, MLP, the F1
> operating point, and the dummy backend are **not** part of it (see §15 for what was removed).
> Where a number is empirical it is marked *approximate - verify against the latest run*.

## Table of contents

1. The problem and the probabilistic objects
2. The target: what counts as a cancellation
3. The temporal split (decision time vs resolution time) and the embargo
4. Features: the roster, every transformation, and leakage control
5. Preprocessing: imputation, scaling, one-hot - and why it differs by model
6. Model 01 - Logistic Regression with ElasticNet
7. Model 02 - XGBoost
8. Model 03 - HistGradientBoosting
9. Model 08 - discrete-time cancellation hazard
10. Probability calibration (isotonic) and why raw scores are not probabilities
11. Cross-validation: temporal vs random
12. Hyperparameter search
13. Evaluation metrics (every one we report), with formulas
14. Decision theory: cost-based threshold, newsvendor quantity, why F1 is wrong here
15. Uncertainty of the per-night forecast: Poisson-binomial, overdispersion, recalibration
16. Explainable AI (permutation importance, PDP/ICE, SHAP)
17. Serving and the app
18. What was removed in v11 and why
19. Glossary (A–Z)

---

## 1. The problem and the probabilistic objects

**Business question.** For every upcoming booking, how likely is it to cancel *before the guest
arrives*? Summed per hotel-night, that gives the **expected number of rooms that will free up**,
which is the basis for deciding how many rooms to oversell.

Three distinct probabilistic objects appear in the project - keep them separate:

- **p = P(cancel before arrival | features)** - a per-booking probability. Models 01/02/03 estimate
  this directly (a *static* classifier: one number per booking).
- **h_d = P(cancel on day d before arrival | survived to day d)** - a per-booking, per-day *hazard*.
  Model 08 estimates this; the cumulative cancel probability by horizon d is then
  `P_cum(d) = 1 − ∏_{u=1..d} (1 − h_u)` (a survival product). This is the *dynamic* view: it tells
  you how the cancel probability evolves as arrival approaches.
- **N_night = Σ (Bernoulli outcomes of that night's bookings)** - the count of rooms that actually
  free up on a hotel-night. Its expectation is `Σ p`; its distribution is **Poisson-binomial** (§15).

A probability is only meaningful for decisions if it is **calibrated**: among all bookings we score
at p ≈ 0.30, about 30% should actually cancel. Ranking quality (AUC) and calibration (Brier) are
different properties; we require both (§10, §13).

## 2. The target: what counts as a cancellation

Built in `00_data_audit.ipynb`; this is the single most consequential modelling choice.

**Modelable rows.** Only *resolved* bookings train the model:
`status ∈ {Canceled, CheckedOut, InHouse}`. Future/Confirmed bookings have no known outcome and are
scored, not trained on.

**Positive event (label = 1):**

```
is_cancelled = (status == "Canceled") AND (cancel_days_before_arrival > 0)
```

where `cancel_days_before_arrival = (arrival − cancellationTime) / 1 day`. In words: the booking was
cancelled **and** the cancellation happened **before arrival**, so a room genuinely freed up in time
to resell or oversell.

**Why the timing condition matters.** A status of "Canceled" alone is not the decision-relevant event.
Two cases are deliberately treated as **non-events (label 0, "the room was not freed in time")**:

- **No-shows** - the guest never arrives but the booking is not a pre-arrival cancellation; the room
  was held, so it did not free up for resale.
- **Post-arrival cancellations** - cancelled after the arrival date; too late to help overbooking.

This makes the label *coherent with the decision*: we predict exactly the thing that creates oversell
capacity. Base rate is **≈ 20.6%** across modelable rows (*approximate*); the test split sits lower
(card base rate ≈ 0.175) because it is the most recent period.

**Target-side column hygiene.** `cancellationTime` and `cancel_days_before_arrival` are *outcome*
fields. They are used to build the label only and are excluded from the feature roster (§4). Drop them
from the modelling frame right after the label is built so they cannot leak.

## 3. The temporal split (decision time vs resolution time) and the embargo

Random splits leak the future into the past: a model validated on randomly held-out rows sees
patterns from booking dates that, in production, would not yet have happened. The split here is
**temporal and two-boundary**, and it is one of the methodologically strongest parts of the pipeline.

Two timestamps matter and they are different:

- **created** = when the booking was made = when we would have to *decide* (decision time).
- **arrival** = when the outcome becomes known (resolution time).

Construction (`00`, split tag `created_quantile_60_15_25`):

```
T1 = arrival.quantile(0.60)     # train | val  boundary (by RESOLUTION time)
T2 = created.quantile(0.80)     # val   | test boundary (by DECISION time)

is_train = arrival  <  T1
is_val   = (created >= T1) & (arrival < T2)
is_test  = created  >= T2
else     -> "embargo"           # excluded from train/val/test
```

Reading it:

- **train** = bookings that had already *resolved* before T1.
- **test** = bookings *decided* after T2 - genuinely "future decisions" relative to the rest, exactly
  the production setting (you score bookings created from now on).
- **embargo** = the straddlers, most importantly bookings *created early but arriving late* (created
  before T1, arriving on/after T1). If those were in train, the model would learn from outcomes that,
  at their decision time, lay in the validation/test future - a subtle leak. Dropping them is the
  **embargo / purge** that gap-based temporal validation requires.

Within train, rows are sorted by `arrival` (mergesort, stable) so that `TimeSeriesSplit` folds are
chronological (§11). Approx sizes (person-level, *verify*): train ~60%, val ~15%, test ~25%, plus a
small embargo set.

### 3.1 Decision time, and what each evaluation actually measures

**Three timestamps — keep them apart (this was previously muddled):**

- **`created`** = when a booking's *features* first exist (information time). It is **not** the overbooking
  decision. Earlier text that called `created` the "decision time" was wrong.
- **decision time** = a horizon **d ≈ 1–14 days before arrival** (rolling). This is when the revenue desk
  actually sets overbooking, and what the model must be good at.
- **`arrival`** = resolution. `outcome_known_date` = `cancellationTime` for a pre-arrival cancel (we learn
  it then), else `arrival`. **The only no-leakage rule: train on outcomes known before the scoring date.**

**Primary, decision-aligned evaluation = horizon-based (the hazard model, §9).** The per-horizon
AUC/AP/Brier at d = 1…14 and the per-night expected-freed bias/coverage measure "how well do we predict
cancellation when a booking is *d days from arrival*" — exactly the overbooking decision. Because the
static models (01/02/03) have no days-until-arrival feature, they emit one fixed probability per booking
regardless of horizon, so they are a per-booking **baseline**, not the decision engine. The hazard model
is horizon-aware and is therefore the primary serving model.

**Secondary, `created`-anchored walk-forward (`src/walkforward.py`).** The rigorous primitive is
`outcome_known_date`. With it:

```
train as-of O : outcome_known_date <= O                  # everything resolved by the cutoff
test  block k : O_k < created <= O_{k+1}  AND  resolved   # the next batch of NEWLY-CREATED bookings
embargo       : created <= O  AND  outcome_known_date > O # in-flight at the cutoff (unavoidable purge)
```

Rolling `O` over 6 bi-monthly steps gives a distribution of metrics, and `assert_point_in_time` enforces
no leakage (all 6 folds pass on the real data). But be precise about what it answers: **"can the model
rank cancellation for newly-*created* bookings?"** — a useful generalisation check, **not** the decision
metric. Do not read its `created` cutoff as a decision time.

**Deployment** fits on **all data resolved by now** (`outcome_known_date ≤ asof`, ≈92% of rows), which is
correct regardless of anchor — nothing is permanently held out, and honesty is preserved because we never
attach a held-out number to weights that trained on that data; we report the procedure's forward metrics
(hazard per-horizon as headline) and confirm with live monitoring. The legacy two-boundary `temporal_split`
is retained only for comparison.

`00` also persists a **run-to-run data summary** (`run_summary.json`) and prints a last-run-vs-this-run
diff, so a data refresh surfaces moved counts/base-rates/date-ranges immediately.

> **Diagnostic pitfall — never cross a `created`-based slice with an `arrival`-based bucket.** A cancel-
> rate-by-*arrival*-quarter heatmap computed on the train+val slice (a `created`-time filter) shows a
> false red edge in the most recent arrival quarters: those quarters survive the filter only via their
> long-lead bookings (short-lead ones were created after the cutoff and are in test), and long lead ⇒
> high cancel. Verified: 2026Q1 in train+val had n≈1.6k, rate≈0.41, median lead 82d, versus n≈17.8k,
> rate≈0.19, median lead 9d on the full resolved data. The fix is to compute time-drift views on **all
> resolved bookings** (or bucket by `created`/decision quarter, which aligns with the split). The model
> and the walk-forward are unaffected — they condition on lead time and bucket on decision time, not on
> arrival quarter.

## 4. Features: the roster, every transformation, and leakage control

**Single source of truth.** `00` writes `Data/feature_roster.json`; every notebook, `src.scoring`,
and the app read feature lists from it (`load_feature_roster`). Hardcoding feature lists anywhere else
is forbidden - that drift is what previously broke scoring. The roster has `numeric`, `categorical`,
`dynamic_numeric`, and an `excluded` map with reasons.

### 4.1 Numeric static features (13)

| feature | transformation | why |
|---|---|---|
| `lead_time_days_log` | `log1p(arrival − created in days)` | lead time is right-skewed; log compresses the tail so it doesn't dominate a linear model. Strong cancel signal: long lead → more time/reason to cancel. |
| `los_nights_log` | `log1p(departure − arrival)` | length of stay, skewed; log for the same reason. |
| `log_gross_amount` | `log1p(total gross €)` | revenue, heavy-tailed; log stabilises variance. |
| `gross_per_night_log` | `log1p(gross / nights)` | price intensity, scale-free-ish after log. |
| `diff_gross_cancellation_fee_log` | `log1p(gross − cancellation fee)` | the € the guest forfeits by *not* cancelling vs cancelling; a refundable booking with a low fee is cheaper to cancel. |
| `adults_n` | numeric count | party size. |
| `arrival_dow` | 0–6 | weekday seasonality of cancellations. |
| `arrival_month` | 1–12 | seasonal demand. |
| `is_weekend_arrival` | 0/1 (dow ≥ 5) | leisure vs business pattern. |
| `is_weekend_arrival`, `has_children`, `has_group`, `has_promo`, `has_corporate_code` | 0/1 flags | behavioural segments; flags avoid sparse one-hots. |

`log1p(x) = ln(1 + x)`: defined at 0, monotone, shrinks large values. For trees it changes nothing
about *ranking* (monotone), but it is kept for parity with the linear model and for readable PDPs.

### 4.2 Categorical static features (7)

`property_name`, `unitGroup_name` (room category), `channelCode`, `ratePlan_category`,
`cancellationFee_name`, `guaranteeType`, `stay_bucket`.

- **`stay_bucket`** = `cut(los_nights, [-1,2,6,365]) → {short, mid, long}`. A coarse, robust view of
  length of stay alongside the continuous `los_nights_log`.
- **`ratePlan_category`** - the raw rate-plan name has hundreds of values (a high-cardinality mess).
  In `00` the raw name is normalised (lowercase, whitespace-collapsed) and mapped to a curated bucket;
  rare/curated-out names collapse to `"other"`. The fitted mapping is persisted in the roster
  (`ratePlan_category_map`) and reused verbatim at scoring so training and serving agree. New, unseen
  rate-plan names map to `"other"` - no recomputation of the rare-collapse on scoring data (which
  would be a parity bug).

### 4.3 Dynamic (scoring-time) features

`days_until_arrival`, `days_since_booking`, `pct_lead_time_elapsed`, `is_within_7d_of_arrival`.
These change every day and only the **hazard model (08)** consumes them, via the time axis. The static
models (01/02/03) do **not** use them.

> **Exact-collinearity flag (acted on in 08).** By definition
> `lead_time = days_until_arrival + days_since_booking`, and `pct_lead_time_elapsed` and
> `is_within_7d` are deterministic functions of those. Feeding all of them plus `lead_time_days_log`
> to one model is rank-deficient. In 08 the time axis is `days_until_arrival`, the baseline scale is
> `lead_time_days_log`, and `days_since_booking` / `pct_lead_time_elapsed` / `is_within_7d` are
> **dropped**.

### 4.4 What is excluded, and why (leakage control)

The roster's `excluded` map is the audit trail. The important categories:

- **Check-in / address leakage** - `guest_country_region`, `primaryGuest_preferredLanguage`,
  `travelPurpose`, `is_international`. These are frequently populated only at check-in. On an upcoming
  booking they are blank, so the model would learn "blank ⇒ behaves like a cancellation" and the
  forecast would explode. Excluded.
- **Company linkage leakage** - `has_company`, `is_repeat_company`, `company_prior_bookings`,
  `company_prior_cancel_rate`. Company linkage is often absent at scoring time. Excluded.
- **Collinear duplicates** - `gross_amount` (use `log_gross_amount`), `ratePlan_type` (coarse parent
  of `ratePlan_category`).

> Leakage re-scan (v11): `cancel_days_before_arrival` / `cancellationTime` are target-side and never
> enter any model's feature matrix. Confirmed clean.

## 5. Preprocessing: imputation, scaling, one-hot - and why it differs by model

All three static models wrap a `ColumnTransformer` + the estimator in a single sklearn `Pipeline`, so
the *exact same* transforms are fit on train folds and applied at validation/test/scoring (no leakage,
no train/serve skew).

**Logistic Regression (01)** - linear models are scale-sensitive and cannot ingest NaN:

- numeric: `SimpleImputer(strategy="median")` → `StandardScaler()`.
- categorical: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`.
- (a vestigial `hist` group exists, constant-0 imputed; currently **empty** after the company
  features were excluded.)

**XGBoost (02)** and **HistGradientBoosting (03)** - trees are scale-invariant and handle missing
values natively, so:

- numeric: **passthrough** (no scaling), NaN handled natively by the tree learner.
- categorical: `OneHotEncoder(handle_unknown="ignore")`.

Why the difference is correct: standardising features for a tree is pointless (splits depend only on
order), and imputing for a learner with a principled missing-value path throws away the information in
"this value is missing." The shipped 02/03 models use **one-hot** categoricals, not XGBoost/HistGB
native categorical handling - so they are **robust to unseen categories** at serving (`handle_unknown
="ignore"` emits an all-zero block). Only the hazard model (08) uses native categorical handling, and
that path needs the dtype-pinning fix (§9).

`StandardScaler`: `z = (x − μ) / σ` with μ, σ learned on the training fold only.
`OneHotEncoder(handle_unknown="ignore")`: each category becomes a 0/1 column; unseen categories at
transform time → all zeros (no crash, no new column).

## 6. Model 01 - Logistic Regression with ElasticNet

**The model.** Logistic regression models the log-odds of cancellation as a linear function of the
features:

```
logit(p) = ln( p / (1−p) ) = β0 + Σ βj xj      ⇒      p = σ(β0 + Σ βj xj),   σ(z) = 1 / (1 + e^−z)
```

- **σ (sigmoid)** maps any real number to (0,1). It is the inverse of the logit.
- **Coefficients βj** are on the **log-odds** scale. `exp(βj)` is an **odds ratio**: a one-unit
  increase in (standardised) `xj` multiplies the odds of cancelling by `exp(βj)`, holding others fixed.
  Because numerics are standardised, βj are comparable in magnitude (effect per 1 SD). For a one-hot
  column, `exp(βj)` is the odds ratio of that category vs the reference.

**Training objective** - minimise penalised log-loss (cross-entropy):

```
L(β) = −(1/n) Σ [ y ln p + (1−y) ln(1−p) ]  +  α · ( ρ·‖β‖₁ + (1−ρ)/2·‖β‖₂² )
```

This is **ElasticNet** regularisation:

- **L1 term `‖β‖₁`** (Lasso): drives some coefficients exactly to 0 → feature selection / sparsity.
- **L2 term `‖β‖₂²`** (Ridge): shrinks coefficients smoothly, handles correlated features gracefully.
- **`l1_ratio = ρ`** ∈ [0,1]: 0 = pure Ridge, 1 = pure Lasso, between = mix.
- **`C = 1/α`**: inverse regularisation strength. Small C = strong regularisation (more shrinkage);
  large C = fit the data harder. Searched on a log scale `loguniform(1e-4, 1e3)`.
- **`penalty="elasticnet"`, `solver="saga"`** - `saga` is the sklearn solver that supports ElasticNet
  on large/sparse data; `max_iter=5000` for convergence.

**Why ElasticNet here.** Many one-hot columns + several correlated numerics. L1 prunes dead categories;
L2 stabilises the correlated survivors. It is the most *interpretable* model in the lineup (read the
coefficients as odds ratios) and the right linear baseline.

## 7. Model 02 - XGBoost

**Gradient-boosted decision trees.** An additive ensemble built stagewise: each new tree fits the
*negative gradient* of the loss (for `binary:logistic`, the residual between label and current
predicted probability), and is added with a shrinkage factor:

```
F_0(x) = const ;   F_m(x) = F_{m-1}(x) + η · f_m(x)
```

Final score is `p = σ(F_M(x))`. XGBoost's objective adds a regularisation term on tree complexity
(number of leaves and leaf weights), and uses second-order (Newton) information.

**Hyperparameters (and what each does):**

- **`n_estimators`** - max number of trees (boosting rounds). Capped high (e.g. 1200) and pruned by
  early stopping.
- **`learning_rate` (η)** - shrinkage per tree. Lower = more trees, better generalisation, slower.
- **`max_depth`** - max tree depth = max interaction order a tree can capture. Higher = more capacity,
  more overfit risk. Searched over {3,4,5,6,8,10}.
- **`min_child_weight`** - minimum summed instance weight (≈ count) in a child before a split is
  allowed. Higher = more conservative, fewer tiny leaves.
- **`subsample`** - fraction of *rows* sampled per tree (stochastic boosting; reduces variance).
- **`colsample_bytree`** - fraction of *features* sampled per tree.
- **`reg_lambda`** - L2 penalty on leaf weights; `reg_alpha` - L1 penalty on leaf weights.
- **`tree_method="hist"`** - histogram-based split finding (bins continuous features → fast).
- **`eval_metric="aucpr"` + `early_stopping_rounds`** - stop adding trees when validation PR-AUC stops
  improving; `best_iteration` is the chosen tree count. Early stopping is the principled way to set
  `n_estimators`.
- **`importance_type="gain"`** - feature importance = total loss reduction attributable to a feature's
  splits (not just split count).

**Why in the lineup.** Captures non-linear effects and interactions (e.g. lead-time × rate-plan) that
the linear model cannot, while staying robust to feature scaling and missingness. Typically the
strongest ranker here.

## 8. Model 03 - HistGradientBoosting

sklearn's histogram gradient boosting - same boosting idea as XGBoost, different implementation,
useful as an independent cross-check (a second strong learner reduces the chance that a quirk of one
library drives the decision).

- **`max_leaf_nodes`** - capacity per tree expressed as leaves (vs XGBoost's depth). Searched ~15–127.
- **`min_samples_leaf`** - minimum samples per leaf; higher = smoother, more conservative.
- **`l2_regularization`** - L2 shrinkage on leaf values.
- **`learning_rate`, `max_iter`** - shrinkage and max number of boosting iterations.
- **`early_stopping=True, validation_fraction=0.1, n_iter_no_change=20`** - HistGB carves its own
  internal validation slice and stops when it plateaus, so `max_iter` is an *upper bound*.
- Native NaN handling (missing values get their own split direction); histogram binning makes it fast.

## 9. Model 08 - discrete-time cancellation hazard

The static models answer "will it cancel?"; the hazard model answers "**when**, and therefore how does
the cancel probability evolve as arrival approaches?" - which is what a daily overbooking desk needs.

**Person-period expansion.** Each booking is expanded into one row per day-before-arrival snapshot on a
grid: daily `d = 1..14` (served at daily resolution, equal width) plus a coarse train-only tail
`{21,30,45,60,90}`. A booking contributes rows for every snapshot it is "at risk" at (it still exists
and has not yet cancelled). The binary outcome `y` for a row is "did it cancel within this window."
Result: ~1.39M person-period rows from ~168k bookings.

**Discrete-time hazard.** On the person-period data we model

```
h_d = P(cancel in window ending at d | survived to d, features)
```

with an XGBoost classifier (`enable_categorical=True`, native categoricals). The **time axis** is
`days_until_arrival`; the **baseline duration scale** is `lead_time_days_log`. This is the
machine-learning analogue of a discrete-time hazard / pooled logistic survival model: by including the
time index as a feature and predicting the per-period event, the classifier *is* a flexible hazard
model.

**Cumulative cancel probability (survival product).** For a booking, the probability it has cancelled
by horizon d is one minus the probability it survived every prior period:

```
P_cum(d) = 1 − ∏_{u=1..d} (1 − h_u)
```

> **The bug that was fixed (critical).** This product must be evaluated on a *fresh forward grid*
> `u = 1..d` for each booking, **not** on the training person-period matrix. The training matrix
> right-truncates cancellers (a booking that cancels at u=3 has no rows for u>3), so multiplying over
> it inverts the cumulative ranking. Scoring on a clean `u=1..d` grid fixes it.

> **The categorical error that was fixed (v11).** XGBoost 2.x recodes the *declared* category array by
> name and raises `Found a category not in the training set` if the scoring frame declares a level the
> model never trained on. The grid had been built with categories taken from the full cleaned data
> (which carried test-only / sub-day-lead rate plans). Fix: pin each categorical to the **exact dtype
> the model trained on** (`g[c] = g[c].astype(tr[c].dtype)`); unseen values become NaN/missing - which
> also makes scoring robust to brand-new hotels and rate plans.

**Per-night expected freed rooms.** For a hotel-night, sum `P_cum(d)` (at the appropriate horizon)
over that night's bookings → expected rooms freed. Validated: per-horizon AUC ≈ 0.71–0.78, well
calibrated; the hazard model beats the static model at every horizon on the matched estimand
(Δ ≈ +0.10 at d=1, +0.049 average for d≤7). Per-night expected-freed tracks actual within +3–8%
(see §15 for the bias and coverage diagnostics).

## 10. Probability calibration (isotonic) and why raw scores are not probabilities

A classifier's raw output (sigmoid of a boosted score, or a regularised logistic score) ranks well but
is usually **not** a calibrated probability: at imbalance and under regularisation, scores are
systematically too high or too low in places. Since we *sum probabilities* to get expected freed rooms,
miscalibration biases the business number directly.

**Isotonic regression** (`CalibratedClassifierCV(method="isotonic", cv=5)`):

- Fits a **monotone non-decreasing** step function g such that `g(raw_score) ≈ P(y=1)`, by the Pool
  Adjacent Violators Algorithm (PAVA). It is non-parametric (more flexible than Platt/sigmoid scaling),
  which suits tree models whose miscalibration is not a simple logistic shift.
- `cv=5`: the calibrator is fit by 5-fold cross-fitting so calibration is learned out-of-fold (no
  leakage from the data the base model saw).
- Trade-off: isotonic can overfit with little data and is piecewise-constant (flat segments). With ~100k
  training rows here that is not a concern.

In the hazard notebook the calibrator is fit explicitly on the validation predictions
(`IsotonicRegression(out_of_bounds="clip")`).

**How we verify calibration:** the **Brier score** (§13) and the **reliability diagram** (bin scores,
plot mean predicted vs observed frequency; the diagonal is perfect). Calibration is gated in model
selection (§14, `best_model` Brier gate).

## 11. Cross-validation: temporal vs random

Two CV strategies are run; only one counts.

- **`TimeSeriesSplit(n_splits=5)` - PRIMARY.** Each fold trains on the past, validates on the
  immediately following block (expanding window). Requires the data to be sorted by time (it is, by
  `arrival`). This respects the arrow of time and is the honest estimate of production performance.
- **`StratifiedKFold(shuffle=True) - DIAGNOSTIC ONLY.** Random folds that preserve class balance.
  Reported *only* to quantify how much the random split over-states performance vs the temporal one
  (the gap is a drift signal). It is never used to select anything.

`cross_validate(..., scoring=["roc_auc","average_precision"])` returns per-fold AUC and AP. The CV
pipeline strips the calibration layer (calibration doesn't change the *ranking* the CV scores measure
and is expensive), which is a deliberate speed trick.

## 12. Hyperparameter search

`RandomizedSearchCV` on the **training set only**, with `cv=TimeSeriesSplit`, `scoring="average_precision"`,
`refit=True`. Random search samples `n_iter` configurations from distributions:

- **`loguniform(a,b)`** for parameters that span orders of magnitude (`C`, `learning_rate`,
  `reg_lambda`, `reg_alpha`, `l2_regularization`) - uniform on the log scale.
- **`uniform(loc, scale)`** for bounded ratios (`l1_ratio`, `subsample`, `colsample_bytree`).
- discrete lists for `max_depth`, `min_child_weight`, `max_leaf_nodes`, etc.

Selection rule (all three notebooks): the search returns `BEST_PARAMS`; then **baseline vs tuned** are
compared on the *validation* set by AP, and the tuned model ships **only if it beats the baseline**
(`model = model_tuned if tuned_AP ≥ baseline_AP else model_base`). Conservative and leakage-free (the
test set is never touched during tuning).

> Honest caveat: "optimal" means *best of {baseline, the searched configs} on validation by AP*. With
> `n_iter=60` over XGBoost's ~7-dimensional space it is a reasonable search, not a global optimum.

## 13. Evaluation metrics (every one we report), with formulas

Notation: TP/FP/FN/TN from a confusion matrix at a threshold; p = predicted probability; y = label.

- **ROC-AUC** `roc_auc_score`. Probability that a random positive is ranked above a random negative.
  Threshold-free, ranking quality. 0.5 = random, 1 = perfect. Robust to imbalance but **optimistic**
  under heavy imbalance (the large negative class makes FPR look small).
- **Average Precision (AP)** `average_precision_score` - area under the Precision–Recall curve,
  `AP = Σ (R_n − R_{n−1}) · P_n`. The **primary ranking metric here** because at ~18–20% prevalence the
  PR curve focuses on the positive (cancellation) class, which is what we act on. Baseline AP = the
  base rate (≈ prevalence), so AP must be read relative to it.
- **Brier score** `brier_score_loss = (1/n) Σ (p − y)²`. Mean squared error of probabilities = a
  **calibration + sharpness** measure. Lower is better. This is the calibration gate in serving.
- **Log-loss / cross-entropy** `−(1/n) Σ [y ln p + (1−y) ln(1−p)]`. The training objective; punishes
  confident wrong probabilities heavily.
- **Precision** `TP/(TP+FP)` - of the bookings we flag as cancel, how many do. **Recall (TPR)**
  `TP/(TP+FN)` - of the cancellations, how many we catch. **F1** = harmonic mean of the two (see §14
  for why we do **not** use it to pick the operating point).
- **Lift@k** - among the top-k% highest-scored bookings, cancel rate ÷ base rate. "How much better than
  random is the top slice." Reported at 5% and 10%.
- **Precision at fixed recall** - precision once you require catching a chosen fraction of cancellations.
- **Confusion matrix** - TP/FP/FN/TN at the chosen threshold; the raw material for the cost calculation.

Which to trust for what: **AP** for ranking/model choice at imbalance, **Brier + reliability diagram**
for calibration, **expected cost** (§14) for the operating decision. AUC is reported but is not the
selection metric.

## 14. Decision theory: cost-based threshold, newsvendor quantity, why F1 is wrong here

Two different decisions, two different rules - do not conflate them.

### 14.1 Per-booking flag - cost-sensitive Bayes threshold

Decide "treat this booking as a likely cancellation" (which permits overselling its room). Costs are
**asymmetric**:

- **False positive** = we oversold expecting a free room, but the guest arrives ⇒ we **walk a guest**.
  `C_walk = 300` (€, configurable).
- **False negative** = we did not oversell, the booking cancels ⇒ the room sits **empty**.
  `C_empty = 80`.

For **calibrated** probabilities, expected cost is minimised by acting iff

```
predict "cancel"  ⇔  p ≥ t*,     t* = C_walk / (C_walk + C_empty) = 300 / 380 ≈ 0.789
```

Derivation: cost of flagging = `(1−p)·C_walk` (pay only if it actually arrives); cost of not flagging =
`p·C_empty`. Flag when `(1−p)·C_walk < p·C_empty` ⇒ `p > C_walk/(C_walk+C_empty)`. Because walking a
guest is ~3.75× worse than an empty room, the threshold is **high (≈ 0.79)** - we only act when fairly
sure. This is exactly the "conservative" rule the revenue team asked for.

In code (`src.scoring`): `analytic_threshold()` returns that ratio; `cost_optimal_threshold()` sweeps
the **validation** predictions to find the threshold minimising `FP·C_walk + FN·C_empty` (the
empirical, calibration-robust version, ≈ 0.75–0.78 on the current models), falling back to the analytic
value when validation predictions are unavailable. The app slider defaults to this and can be overridden.

**Why not F1?** F1 = `2·P·R/(P+R)` weights precision and recall **equally** and ignores both true
negatives and the cost ratio. Under 300/80 costs, the F1-optimal threshold (~0.24) flags ~25–30% of
bookings; on held-out data that is **~3.6–4.5× more expensive** than the cost threshold (validated on
01/02/03 predictions). F1 is a balanced-class convenience metric; it is the wrong objective for an
asymmetric-cost decision.

### 14.2 Per-night oversell quantity - newsvendor critical fractile

How many rooms to oversell on a hotel-night is a *quantity* decision, not a per-booking flag. The
newsvendor solution is to oversell up to the **critical fractile** of the freed-rooms distribution:

```
q* = C_u / (C_u + C_o)
```

With the project's numbers (`C_u = 80` underage = an empty room you could have filled; `C_o = 300`
overage = walking a guest) `q* = 80/380 ≈ 0.21`. Because 0.21 < 0.5, you oversell **below** the mean
expected freed rooms: `oversell ≈ E[freed] + z·SD`, `z = Φ⁻¹(0.21) ≈ −0.81` - i.e. a conservative
quantity. The hazard notebook (08) computes this; the app's per-day recommendation implements the
conservative spirit by counting only high-confidence cancellations (p ≥ slider threshold), gated by an
occupancy floor (only oversell when near full).

> Both numbers (0.79 and 0.21) are "conservative" but answer different questions; 0.79 is the
> per-booking flag threshold, 0.21 is the per-night quantity fractile. Keep them separate.

## 15. Uncertainty of the per-night forecast

Two issues with the per-night expected freed rooms, both checkable with the diagnostic cells added to
notebook 08.

**Aggregate bias (+3–8%).** Even with calibrated per-booking probabilities, the per-night *sum* can
drift a few percent (isotonic fit on validation; survival product compounds small per-step errors).
Fix/check: fit a single scalar `r = Σactual / Σexpected` on validation nights and apply to test; if the
test bias collapses and `|bias_recal| < |bias_raw|`, the one-parameter correction generalises. (A
Poisson GLM with `expected` as offset is the equivalent multiplicative fit.)

**Interval coverage (87–90% vs nominal 95%).** The per-night total is a sum of Bernoullis; **if**
cancellations were independent its variance is the **Poisson-binomial** variance `Σ p(1−p)`, giving the
interval `E ± 1.96·√(Σ p(1−p))`. Observed coverage **below** nominal means independence is too
optimistic - real cancellations are **positively correlated** within a night/property (corporate blocks
cancel together, events, weather) and `p` is itself estimated. Diagnose with the **overdispersion
factor**

```
φ = mean_over_nights [ (actual − expected)² / Σ p(1−p) ]
```

`φ ≈ 1` ⇒ independence holds; `φ > 1` ⇒ inflate the SD by `√φ`. The diagnostic cell prints coverage at
50/80/90/95% before and after `√φ` inflation; if the inflated coverage lands on nominal, overdispersion
is confirmed and the planning bands should use `√φ`-inflated intervals (or a group/cluster bootstrap
that resamples bookings by night/property).

## 16. Explainable AI

- **Permutation importance** - shuffle one feature's column and measure the drop in a chosen score
  (here AP/AUC). Model-agnostic, measures *predictive* contribution on held-out data (unlike tree
  "gain", which is in-sample and biased toward high-cardinality features). HistGB uses this (no native
  gain). Caveat: correlated features share/steal importance.
- **Partial Dependence Plot (PDP)** - average predicted probability as one feature is swept across its
  range, marginalising the rest. Shows the *average* shape of a feature's effect (e.g. cancel
  probability rising with lead time). Assumes feature independence; misleading under strong correlation.
- **ICE (Individual Conditional Expectation)** - the same sweep but one line per booking; the PDP is the
  average of the ICE lines. Divergent ICE lines reveal interactions the PDP averages away.
- **SHAP (Shapley Additive exPlanations)** - attributes each prediction to its features using Shapley
  values from cooperative game theory: the unique attribution that is locally accurate (contributions
  sum to the prediction minus the base value) and consistent. `TreeExplainer` computes them exactly and
  fast for tree models. Use the waterfall for a single booking (why *this* prediction), the beeswarm for
  global structure (which features matter and in which direction), and dependence plots for a feature's
  effect coloured by an interacting feature. (Method: Lundberg & Lee, 2017 - verify the exact reference
  if you cite it.)

## 17. Serving and the app

- **`src.scoring.score_upcoming(model_name, threshold=None)`** - loads a trained pipeline, pulls
  upcoming bookings, rebuilds the roster features with `build_features` (which must mirror `00`), scores,
  and emits `cancel_proba`, `pred_cancel` (at the cost-optimal or overridden threshold), `risk_bucket`,
  and `model_used`.
- **`best_model()`** - picks the serving model: highest test **AP** among models whose test **Brier** is
  within `0.005` of the best (a calibration gate, so we never ship a sharp-but-miscalibrated ranker).
- **`serving_thresholds(name)`** → `(low, high)`: `high` = cost-optimal validation threshold,
  `low` = validation base rate (below-average-risk display band).
- **App** (`dash_app`): a backend **facade** (`dash_app/backend/__init__.py`) gives every page one
  interface and one canonical schema (`schema.COLUMNS`), so the model is swappable from a dropdown and
  the threshold from a slider with no page changes. The threshold is a **post-scoring** derivation
  (probabilities don't change), so moving the slider re-derives counts and the conservative
  recommendation without re-running the model. New hotels are picked up automatically (units from the
  performance table; unseen categories handled by `handle_unknown="ignore"`), which is the scalability
  requirement.

### 17.1 Retraining (`src.training`, app-callable)

Fit logic lives in `src.training` so the notebooks (thin drivers) and the app call the same code.
`build_pipeline` reconstructs each model's exact preprocess+estimator+calibration stack (§5–§8).

- **`retrain(model, mode="refit")`** — read the **frozen** hyperparameters from the model card, fit the
  pipeline on **all data resolved by `asof`** (the deployment set, ~92% of rows), recompute the cost
  threshold, persist the joblib + an updated card. The routine, fast retrain.
- **`retrain(model, mode="retune")`** — re-run `RandomizedSearchCV` (temporal CV, AP) first, then refit
  on all resolved data. Heavier; use when the feature set or data distribution has shifted materially.
- **`walk_forward_eval(model)`** — the §3.1 forward-metric distribution; `retrain` embeds it in the card
  so the deployed model carries an honest performance estimate (confirmed by live monitoring).
- **`select_models()`** → `{primary: "hazard", static: <best>}`. The **hazard model is primary**
  (horizon-aware); `retrain("hazard", …)` dispatches to `src.hazard.retrain_hazard`, which does the
  person-period fit (HP grid + isotonic) on all resolved bookings and persists
  `Data/08_hazard_model.joblib` + a card. `src.hazard.score_upcoming_hazard` serves per-booking
  P(cancel before arrival) via the survival product (fresh u=1..D grid, train-dtype-pinned categoricals).
  The best static model is the per-booking baseline.

**Calibration under refit.** Because the pipeline wraps the estimator in `CalibratedClassifierCV(cv=5)`,
refitting on the full set still cross-fits the calibrator out-of-fold — so deployment calibration is not
contaminated by the larger training set.

**Retraining guards (the "add a column safely" machinery).** A feature is added by simply un-excluding a
raw, booking-time column in `00`'s blocklist; it then flows through the roster into every notebook and
the scorer automatically. On retrain `src.training` enforces:

- `roster_fingerprint` — a hash of the sorted feature set, stored in the card.
- `feature_change_report` — diffs the current roster against the deployed model and logs added/removed
  columns (additions never error — they just enter the next fit).
- **mode policy** — if the feature set changed and you asked for `refit`, it *warns* that the frozen
  hyperparameters were tuned for the old set and recommends `retune`.
- `scoring_null_audit` — per-feature null-rate on the upcoming frame; flags any feature that is
  ~always-blank on upcoming bookings (the check-in/company-leakage signature), so a bad un-exclude is
  loud, not silent.
- **serving staleness** — if the deployed model's feature signature ≠ the current roster, scoring raises
  a clear "retrain" message instead of a cryptic sklearn shape error.

Statistically: a *redundant* (perfectly collinear) column adds no information and cannot improve any
model; a *correlated-but-not-identical* column can help the trees (and muddy their importances) while
the logreg L1 penalty tends to zero one of the collinear set. Add it, let `walk_forward_eval` show
whether AP/Brier/cost improve, and drop it if not.

## 18. What was removed in v11 and why

- **RandomForest, MLPClassifier** - out of the lineup; not registered, not served. (RF: weaker
  calibrated ranker here; MLP: tuning/calibration cost not justified by performance.)
- **F1 operating point** - replaced by the cost-based threshold (§14), which is now the **primary**
  operating point persisted in 01/02/03 (`cost_optimal`), with `f1_optimal` kept only as a visible
  reference line. `src.scoring` derives the threshold from the validation predictions.
- **"`created` = decision time" framing** - corrected (§3.1). `created` is information time; the
  overbooking decision is d≈1–14 days before arrival. The **hazard model (08) is now the primary,
  horizon-aware serving engine**, persisted + retrainable via `src.hazard`; the static models are a
  per-booking baseline. The `created`-anchored walk-forward is a secondary new-booking generalisation
  check, and the single frozen `temporal_split` is kept for comparison only.
- **Fit logic inside notebooks** - moved to `src.training` (§17.1); notebooks 00→01/02/03→05 are now thin
  drivers, so the dash app can retrain through the same code.
- **Dummy backend** - synthetic data path; being removed so the app runs on real models/data only with a
  graceful empty-state (still pending).
- The old 1600-line `pipeline_reference.md` is archived at
  `reports/_archive/pipeline_reference_pre_v11.md` (kept for history; safe to delete).

**Added in v11:** `outcome_known_date` + `src/walkforward.py` (point-in-time folds, run-to-run diff);
`src/training.py` (`retrain` refit/retune, `walk_forward_eval`, `select_models`, roster/leakage guards).

## 19. Glossary (A–Z)

- **Average Precision (AP)** - area under the precision–recall curve; primary ranking metric at imbalance.
- **Base rate / prevalence** - share of positives (cancellations) ≈ 20.6% (test ≈ 17.5%).
- **Bayes-optimal threshold** - the threshold minimising expected cost for calibrated probabilities,
  `C_walk/(C_walk+C_empty)`.
- **Boosting** - additive ensemble; each tree fits the current residual/gradient.
- **Brier score** - mean squared error of probabilities; calibration + sharpness; lower better.
- **Calibration** - predicted probabilities match observed frequencies.
- **`CalibratedClassifierCV`** - sklearn wrapper applying isotonic/sigmoid calibration via cross-fitting.
- **Censoring** - outcome not observed as an event in the window (here: no-shows / post-arrival cancels
  treated as survived/0).
- **`colsample_bytree`** - fraction of features sampled per boosting tree.
- **`ColumnTransformer`** - applies different preprocessing to different column groups.
- **Critical fractile** - newsvendor quantile `C_u/(C_u+C_o)` for the oversell quantity (≈ 0.21).
- **Cross-entropy / log-loss** - logistic training objective.
- **Discrete-time hazard** - per-period conditional event probability `h_d`; cumulative via survival
  product.
- **Early stopping** - halt boosting when validation metric stops improving; sets effective tree count.
- **ElasticNet** - L1+L2 penalty; `l1_ratio` mixes, `C` sets strength.
- **Embargo / purge** - rows dropped between train and test to prevent decision/resolution-time leakage.
- **Gain (importance)** - total loss reduction from a feature's splits (in-sample, biased).
- **`handle_unknown="ignore"`** - OHE emits all-zeros for unseen categories (robust serving).
- **Hazard** - see discrete-time hazard.
- **HistGradientBoosting** - sklearn histogram-binned gradient boosting; native NaN handling.
- **Isotonic regression** - monotone non-parametric calibration map (PAVA).
- **Leakage** - using information unavailable at decision time; here check-in/address/company fields and
  outcome timestamps.
- **`learning_rate` (η)** - boosting shrinkage per tree.
- **Lift@k** - top-k% cancel rate ÷ base rate.
- **`log1p`** - `ln(1+x)`; skew-damping transform used on lead time, LOS, gross, etc.
- **Logit** - `ln(p/(1−p))`; linear-model output scale.
- **`max_depth` / `max_leaf_nodes`** - tree capacity (interaction order / leaf count).
- **`min_child_weight` / `min_samples_leaf`** - minimum mass per leaf; higher = more conservative.
- **Newsvendor** - inventory model giving the cost-optimal oversell quantity via the critical fractile.
- **Odds ratio** - `exp(β)`; multiplicative effect on odds in logistic regression.
- **One-hot encoding** - categorical → 0/1 indicator columns.
- **Overdispersion (φ)** - ratio of observed to independence-model variance; >1 ⇒ correlated outcomes.
- **Permutation importance** - out-of-sample importance via column shuffling.
- **Person-period grid** - one row per booking per day-before-arrival snapshot (hazard model input).
- **Platt scaling** - sigmoid (parametric) calibration alternative to isotonic.
- **Poisson-binomial** - distribution of a sum of independent non-identical Bernoullis; variance
  `Σ p(1−p)`.
- **`reg_lambda` / `reg_alpha` / `l2_regularization`** - L2 / L1 penalties on leaf weights.
- **Reliability diagram** - predicted vs observed frequency by bin; the calibration plot.
- **ROC-AUC** - threshold-free ranking quality.
- **`saga`** - sklearn solver supporting ElasticNet on large data.
- **SHAP** - Shapley-value local+global attribution; `TreeExplainer` for tree models.
- **Sigmoid (σ)** - `1/(1+e^−z)`; inverse logit.
- **`StandardScaler`** - `(x−μ)/σ`; needed for the linear model, not for trees.
- **`subsample`** - fraction of rows per boosting tree.
- **Survival product** - `P_cum(d) = 1 − ∏(1−h_u)`.
- **Temporal split** - chronological train/val/test by created (decision) and arrival (resolution) time.
- **`TimeSeriesSplit`** - chronological expanding-window CV (primary).
- **XGBoost** - gradient-boosted trees; the strongest static ranker here and the hazard learner.
