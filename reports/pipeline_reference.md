# Overbooking model pipeline — mental map (v11)

The one-page picture of what runs and why. For the math and every parameter, see the companion
**`statistical_manual.md`**. Deliberately short: anyone should grasp the whole pipeline in ten minutes.

## What it does

For every **upcoming booking** we estimate **P(cancel before arrival)**. Summed per hotel-night that is
the **expected number of rooms that will free up**, which drives a **conservative overbooking
recommendation** per hotel per day. Runs across all STAYERY locations; new hotels appear automatically.

```
BigQuery reservations
  └─ 00_data_audit : clean, build the coherent target, engineer features (+ §4.3 log cols),
                     write feature_roster.json, add outcome_known_date, persist walk-forward folds,
                     run-to-run data diff
       ├─ 01 logreg ─┐
       ├─ 02 xgboost ─┼─ thin drivers over src.training: fit → tune (temporal CV, AP) → isotonic
       ├─ 03 histgb ─┘                       calibrate → persist joblib + model_card + predictions
       └─ 08 hazard : person-period (daily d=1..14 + tail) → survival product → per-night expected freed
  └─ src.scoring / src.hazard : score ALL upcoming bookings → cost-based threshold → risk buckets
  └─ dash_app : hazard-served recommendation + threshold slider + retrain buttons
```

## The target (the one choice everything hangs on)

Positive (1) iff `status == "Canceled"` **and** `cancel_days_before_arrival > 0` — it cancelled *before*
arrival, freeing a room in time. **No-shows and post-arrival cancels are 0** (censored "arrived"). Only
resolved bookings train. The label is identical to the thing we act on. Base rate ≈ 20%.

## Features

`Data/feature_roster.json` is the single source of truth (numeric + categorical + dynamic). Skewed
money/time features are `log1p` (`lead_time_days`, `los_nights`, `gross_per_night`,
`diff_gross_cancellation_fee`, `gross_amount` — 00 §4.3). **Excluded for leakage:** check-in / address /
company fields (blank at scoring) and collinear duplicates. Outcome fields (`cancellationTime`,
`cancel_days_before_arrival`, `outcome_known_date`) are target/split metadata — never features.

## The split — arrival-anchored walk-forward (`src/walkforward.py`)

We do **not** decide at booking creation; we decide overbooking for bookings **arriving in the next ~14
days**. So validation is arrival-anchored. The primitive is `outcome_known_date` (when a label became
known: `cancellationTime` for a pre-arrival cancel, else `arrival`). At each scoring date **S** (rolled
~monthly over the last year):

- **train** = every booking **resolved by S** (`outcome_known_date ≤ S`) — *all* resolved data.
- **test** = bookings **active at S, arriving within the horizon** (`created ≤ S < arrival ≤ S+14`, not
  yet resolved) — exactly the population the desk decides on; graded on cancel-before-arrival.

`assert_point_in_time` hard-guarantees no fold leaks. The reported metric is the *distribution* of
one-step-ahead AUC/AP/Brier/cost across folds (the procedure's honest performance, confirmed by live
monitoring). Three timestamps stay distinct: **created** = info time, **decision** = ~14d before
arrival, **arrival** = resolution. (`created` is never the "decision time".)

## The models

| # | model | role |
|---|---|---|
| **08** | **Discrete-time hazard (XGBoost)** | **PRIMARY engine** — horizon-aware P(cancel before arrival) → per-night expected freed rooms (`src/hazard.py`) |
| 01 | Logistic Regression (ElasticNet) | horizon-blind per-booking **baseline** (interpretable; odds ratios) |
| 02 | XGBoost | per-booking baseline / strong ranker |
| 03 | HistGradientBoosting | independent cross-check baseline |

The static models give **one fixed probability per booking** (no days-until-arrival feature) → a
baseline, not the decision engine. The **hazard model** is time-resolved: person-period grid (daily
**d = 1..14** + coarse tail), learns `h_d = P(cancel in window d | survived)`, serves
`P(cancel before arrival) = 1 − ∏(1−h_u)` over each booking's remaining days (the survival product;
categoricals pinned to the train dtype). All models are **isotonic-calibrated** so probabilities can be
summed safely. **Out of the pipeline:** RandomForest, MLP, AFT/RSF, the F1 operating point, the dummy
backend.

## Calibration + per-night aggregation (statistically correct)

Per hotel-night: **expected freed = Σ p** with Poisson-binomial variance **Σ p(1−p)**. A one-parameter
aggregate recalibration (`r = Σactual/Σexpected` fit on validation) corrects residual bias; the
**overdispersion factor φ** (>1 ⇒ correlated cancellations) inflates the interval by `√φ`
(`src.hazard.per_night_table` / `recalibration_factor` / `coverage_report`).

## The decision — cost-based, not F1

Asymmetric costs: **walk a guest = 300 €**, **empty room = 80 €**. For calibrated probabilities the
Bayes-optimal flag threshold is `300/(300+80) ≈ 0.79` (conservative — act only when ~79% sure). The
per-night oversell *quantity* uses the newsvendor critical fractile `80/(80+300) ≈ 0.21` (oversell below
the mean). `cost_threshold_from_scores` is the single shared definition (notebooks + scoring + app).
**F1 is gone** — it ignores the cost asymmetry (≈4× costlier on held-out data).

## Production scoring — ALL upcoming bookings

`src.scoring.score_upcoming()` loads `load_reservations(upcoming_only=True)` = **every** booking with
`arrival ≥ now`, builds features, and scores them — independent of train/test membership (those only
governed historical evaluation). `apply_scoring_bounds` drops only rows with uncomputable features,
never on status. When a hazard model is on disk, `real.get_scored_bookings` overrides `cancel_proba`
with the hazard survival score on that same full set. So every live Confirmed/unresolved booking is
evaluated.

## src modules + retrain API

`src.training.retrain(model, mode="refit"|"retune")` fits on all resolved data (refit = frozen HP;
retune = re-search), with guards: a roster fingerprint, a feature-change report (adding a column flows
in + is logged; refit on a changed feature set warns to retune), and a scoring-time null audit
(flags ~always-blank-on-upcoming features = leakage smell). `select_models()` → `{primary: hazard,
static: best}` (best = highest AP within a Brier tolerance). `walk_forward_eval` (static) and
`hazard.walk_forward_eval_hazard` (matched, apples-to-apples: same arrival-anchored rows + label;
promotion signal = mean(ΔAUC) > its own std, not a noise gate).

## The app (real-only)

One backend **facade**; the dummy backend is fully removed. Locations from `configs/locations.yaml`
(`locations.py`). **Overbooking page**: hazard-served `cancel_proba`, a **threshold slider** (default =
cost-optimal on validation) driving the conservative per-day recommendation. **Model & Performance
page**: real ROC / PR / calibration / confusion computed from the persisted test predictions.
**Datenaktualisierung page**: **Refresh** (re-pull + re-score) and **Retrain (refit / retune)** buttons
(`B.retrain_models`). Graceful empty-state when no model/data is on disk.

## Run order (overnight)

`00 → 01, 02, 03 → 05 → 08`. `00` rebuilds data/roster/folds; `01/02/03` retrain + persist cards +
predictions; `05` compares; `08` persists the hazard model + runs the matched comparison + per-night
calibration. Then the app serves real models with the hazard model primary.

## v11 changelog

Coherent cancel-before-arrival target; `_log` features; **arrival-anchored walk-forward**; hazard model
primary + persisted/serveable/retrainable (`src/hazard.py`); cost-based threshold replacing F1;
per-night aggregation + recalibration + coverage; RandomizedSearch + early stopping (hazard); densified
1–14 grid; survival-product + categorical-dtype bugfixes; `src.training` retrain API + guards;
notebooks as thin drivers; **dummy backend fully removed**, real-only app with hazard serving, real
model-performance curves, threshold slider, retrain buttons; run-to-run data diff in 00.
