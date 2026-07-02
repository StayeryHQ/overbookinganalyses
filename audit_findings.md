# Phase 1 — Codebase Audit Findings

Generated: 2026-07-02. Scope: read-only audit (no app code written). Read this,
then confirm the open questions in the final section before I start Step 2.

## How this audit was produced (and its limits)

- All findings come from reading the source, the JSON/CSV artifacts, and the two
  cached Parquet files directly.
- **I could NOT run the models or hit BigQuery** in the audit sandbox: `scikit-learn`,
  `xgboost` and live BigQuery credentials/network are not available here. Model
  format, inputs and outputs below are read from the source code and the persisted
  `model_card.json` files — not from un-pickling the artifacts. Flagged inline where
  it matters. Everything should be re-verified once the project venv is used.
- Row counts and column lists are exact (read from Parquet metadata). Model metrics
  are quoted verbatim from the model cards.

---

## 1. Codebase location & structure

Everything lives in the project root (`OverbookingAnalyse/`). It is a `uv`-managed
Python 3.12 package, not a set of loose notebooks.

- `src/` — the installable package (`import src`):
  - `data_loader.py` — BigQuery pulls, PII strip, dtype coercion, Parquet cache.
  - `scoring.py` — model registry, loading, cost-based thresholds, `score_upcoming()`.
  - `hazard.py` — the discrete-time hazard model (train / load / score).
  - `features.py`, `walkforward.py`, `training.py`, `utils.py`, `paths.py`.
- `notebooks/` — `00_data_audit`, `01_logreg`, `02_xgboost`, `03_histgb`,
  `05_model_comparison`, `08_hazard`, `09_live_scoring`, plus `experiments/`.
- `dash_app/` — **skeleton only**: empty `__init__.py`, `components/__init__.py`,
  brand fonts under `assets/fonts/`, and `backend/locations.py` which is a
  placeholder returning `[]`. There is **no actual Dash app yet** — consistent with
  Step 2 being "app foundation".
- `configs/` — `paths.yaml`, `stayery_brand.yaml`, `reservations_schema.json`.
- `Data/` — Parquet caches, two model `.joblib` files, JSON metadata (git-ignored).
- `reports/` — `tables/` (model cards, audit CSVs), `figures/`.
- `main.py` — CLI (`refresh` / `score` / `status`). `verify_pipeline.py` — leakage/
  consistency checks. `pyproject.toml` + `uv.lock` — dependencies.

**Leftover cruft (not blocking, worth cleaning in a later phase):**
- `.streamlit/config.toml` exists and references mirroring a *"RevenueBlindSpots"*
  Streamlit app. The decided framework is Dash, so this is a remnant of an earlier
  direction.
- `src/paths.py`'s docstring references `src/revenueblindspots/paths.py` — a
  copy-paste remnant from that other project.

---

## 2. BigQuery connection & authentication

**Authentication — already matches the target pattern (good news).**
Both query helpers create the client with a bare `bigquery.Client()`
(`src/data_loader.py`, `_query_bigquery` and `_query_property_performance`). A bare
client uses **Application Default Credentials (ADC)**, which is exactly the
env-driven pattern the project instructions ask for:

- Locally it uses the developer's `gcloud auth application-default login` session.
- In a container it uses the service-account key referenced by
  `GOOGLE_APPLICATION_CREDENTIALS`.
- Both are consumed transparently by ADC with **no code branching**.

I grepped the whole tree for hardcoded key paths / `GOOGLE_APPLICATION_CREDENTIALS`
/ `from_service_account*` and found **none**. So the doc's concern about a hardcoded
local key path does **not** apply — no change needed there.

- Minor: `BQ_PROJECT`/`BQ_DATASET`/`BQ_TABLE` are hardcoded module constants
  (`stayery-analytics` / `reporting` / `reservations`). These are identifiers, not
  secrets, but if you want them env-overridable that would be a small Step 2 add.
- The exact `bigquery.Client()` initialization call should still be checked against
  the current `google-cloud-bigquery` docs during Step 2 (as the instructions
  require) — I did not verify the signature against live docs in this audit.

**SQL queries — they exist, but as Python-built strings, not `.sql` files.**

1. `reservations` (`_query_bigquery`):
   `SELECT * EXCEPT(<PII columns present in the live schema>) FROM
   \`stayery-analytics.reporting.reservations\`` (+ optional `LIMIT`).
   The PII exclusion list is intersected with the live table schema at query time,
   so a renamed/missing PII column can't break the query. Returns the full booking
   history, PII-stripped at SQL level; a `strip_pii()` pandas pass is a second safety
   net; then dtypes are coerced and the result is cached to Parquet.

2. `property_performance_daily` (`_query_property_performance`):
   `SELECT <operational columns present> FROM
   \`stayery-analytics.reporting.property_performance_daily\``.
   Only an **allow-list of operational columns** is selected (occupancy + counts);
   **all revenue columns are deliberately never listed** → never pulled. Same
   drift-robust intersection with the live schema.

**Caching** already follows the instructions' "don't query live on every
interaction" rule for reservations: a `reservations_raw_no_pii.parquet` cache with a
`force_refresh` flag. The same pattern is coded for property performance but no
cache has ever been built (see §5).

---

## 3. Data dictionary

### 3a. `reservations` (BigQuery `reporting.reservations`)

- Full BQ schema (`configs/reservations_schema.json`): **100 columns**.
- After excluding **27 PII columns** at SQL level → the cached raw Parquet
  (`Data/reservations_raw_no_pii.parquet`) has **73 columns × 216,726 rows**.
- `status` distribution in the raw cache: CheckedOut 163,354 · Canceled 45,383 ·
  Confirmed 5,074 · NoShow 2,236 · InHouse 679. (The 5,074 `Confirmed` rows are the
  upcoming/open bookings that scoring targets.)
- **11 properties** (matches the brief), `property_name` values: Berlin
  Friedrichshain, Bielefeld Hauptbahnhof, Bochum Ehrenfeld, Bremen Am Wall, Cologne
  Ehrenfeld, Cologne Sülz, Frankfurt Sachsenhausen, Fürth Hallstraße, Gütersloh
  Langer Weg, Osnabrück Johannisstraße, Wolfsburg Schachtweg.

Selected raw columns (name · inferred dtype · example value):

| column | dtype | example |
|---|---|---|
| `id` | object (str) | `'SQTLDGDQ-1'` |
| `bookingId` / `blockId` / `groupName` | object | grouping keys, mostly null |
| `status` | object | `'InHouse'` (raw string; encoded to 0/1 only in the clean set) |
| `property_name` | object | `'Bochum Ehrenfeld'` |
| `channelCode` | object | `'ChannelManager'`, `'Booking.com'`, `'Direct'`, … (19 values) |
| `source` | object | OTA source, used to fill `channelCode` |
| `totalGrossAmount_amount` | float64 | `1696.32` |
| `cancellationFee_fee_amount` | float64 | forfeit fee if cancelled |
| `adults` | Int64 | `2` |
| `children` / `childrenAges` | Int64 / object | party composition |
| `unitGroup_name` | object | `'BIG'`, `'BIGGER'`, `'BOX'` (10 values) |
| `ratePlan_name` | object | free-text rate-plan name (bucketed downstream) |
| `guaranteeType` | object | `'CreditCard'` / `'Prepayment'` / `'Company'` |
| `arrival` / `departure` / `created` / `modified` / `cancellationTime` | datetime64[us, UTC] | `2026-06-13 11:34:41+00:00` |
| `ratePlan_isSubjectToCityTax` | boolean | `True` / `False` |
| `guest_id` / `is_first_res` / `is_last_res` | Int64 | repeat-guest flags |

Full column list is in the raw Parquet; the remaining columns are mostly IDs/codes
(`property_id`, `ratePlan_id`, `unit_*`), fee/commission fields, and company fields.

**Also present — the cleaned modeling dataset** (`Data/reservations_clean.parquet`):
**179,349 rows × 31 columns**, produced by `00_data_audit.ipynb`. This is the
feature-engineered, leakage-audited table the models train on. A complete,
per-column data dictionary for it already exists at
`reports/tables/00_audit/data_dictionary.csv` (role, description, cardinality,
missingness, top values). Key facts: base rate ≈ **19.8%** positive
(`status` = cancel at/before arrival, int8 0/1); arrivals span **2022-08-01 →
2026-06-27**. *(Note: only ~1 year of pre-2023 data exists, so the later
"pre/post-COVID" split in the Cancellation-History phase will likely be thin — flag
for that phase, not now.)*

### 3b. `property_performance_daily` (BigQuery `reporting.property_performance_daily`)

**⚠ I cannot give real sample values, dtypes-from-data, or a row count for this
table** — there is no local cache (`Data/property_performance_daily.parquet` does
not exist) and no live BigQuery access in the audit sandbox. The following is the
**intended** schema, read from the allow-list and dtype coercion in
`data_loader.py`. Verify against the live table when creds are available.

| column | intended dtype | meaning |
|---|---|---|
| `businessDay` | datetime64[UTC] | one row per property per business day |
| `propertyId` | string | property code (**note: `Id`, not the `property_name` used in reservations — a join/mapping will be needed**) |
| `houseCount` | Int64 | bookable units (used as the property's unit count) |
| `soldCount` | Int64 | rooms sold |
| `outOfOrderCount` | Int64 | OOO rooms |
| `arrivalsCount` | Int64 | arrivals |
| `departuresCount` | Int64 | departures |
| `noShowsCount` | Int64 | no-shows |
| `cancellationsCount` | Int64 | cancellations |
| `occupancyPercentage` | float64 | occupancy % |

Revenue columns exist in the table but are intentionally excluded.

### 3c. Table count vs. the brief

The brief expected **two** tables — I found exactly two referenced
(`reporting.reservations` and `reporting.property_performance_daily`). ✅ No table-count
discrepancy. (The model/discrepancy issue is in §4/§6.)

---

## 4. Model artifacts

**Only TWO model files exist on disk**, both joblib-serialized:

| file | size | what it is |
|---|---|---|
| `Data/02_xgboost_model.joblib` | 9.3 MB | static XGBoost classifier (sklearn Pipeline) |
| `Data/08_hazard_model.joblib` | 1.8 MB | discrete-time hazard model (dict artifact) |

There is **no** `01_logreg`, **no** `03_histgb`, and **no** baseline/mean model file
present, even though the XAI page (later phase) and `main.py` reference them.

### `08_hazard_model.joblib` — this is "model 8" by numbering

- **Format:** joblib. The persisted object is a **dict**, not a bare estimator:
  `{model: XGBClassifier, iso, iso_bands, num, cat, cat_dtypes, snap, axis, hp,
  val_ap, ...}` (per `src/hazard.py`).
- **Inputs:** 18 numeric + 7 categorical roster features **plus the day axis
  `days_until_arrival`** (`features_numeric`/`features_categorical` in its model card).
- **Output:** NOT a single probability from the raw estimator. It's a discrete-time
  **hazard** `h_d = P(cancel in window ending d days before arrival | survived)`,
  per-band isotonic-calibrated, then turned into a per-booking
  **P(cancel before arrival)** via the survival product
  `1 − Π(1 − h_s)` over the trained snapshot grid (daily 1–14 + coarse tail to 270).
  Served by `src/hazard.py::score_upcoming_hazard`, **not** by `scoring.py`.
- **Reported quality:** `val_ap ≈ 0.075` (average precision on the validation
  person-period grid; from its card). This is low and not directly comparable to the
  static model's booking-level AP.

### `02_xgboost_model.joblib` — the model the serving code actually uses

- **Format:** joblib; a fitted sklearn **Pipeline** (scoring calls
  `pipeline.predict_proba(X)[:, 1]`).
- **Inputs:** 14 numeric + 7 categorical (its card). The 14 vs. the roster's 18
  numeric is expected: the tree family uses the raw skewed columns and drops the 4
  `*_log` twins (confirmed by `verify_pipeline.py` §7 and the roster's `log_twins`).
- **Output:** calibrated **P(cancel at/before arrival)** per booking.
- **Reported quality (walk-forward, from card):** AUC ≈ 0.758, AP ≈ 0.284,
  Brier ≈ 0.110, cost-optimal threshold ≈ 0.789 (with the default asymmetric costs
  walk €300 / empty room €80).

### The registry disagrees with what's on disk

`src/scoring.py::MODEL_REGISTRY` registers only `logreg` (01), `xgboost` (02),
`histgb` (03) — and **not** the hazard model. Since only `02` exists on disk:
- `list_available_models()` → `['xgboost']` only.
- `best_model()` / `best_model_by_auc()` → `xgboost`.
- The hazard model (08) is **unreachable** through `scoring.load_model()`; it has a
  separate loader (`hazard.load_hazard`).

---

## 5. The non-functional `property_performance_daily` loader — diagnosis

The intended function is **`load_property_performance()`** in `src/data_loader.py`
(with `_query_property_performance()` doing the SQL and `property_universe()`
wrapping it to build the per-property unit list). Precise reasons it does nothing
useful today:

1. **No cache + needs a live connection.** There is no
   `Data/property_performance_daily.parquet`, so every call falls straight through to
   `_query_property_performance()`, which needs live BigQuery (ADC creds + network).
   Offline/without creds that raises.
2. **Never wired in / not in the public API.** `load_property_performance` and
   `property_universe` are **not exported** in `src/__init__.py` (only
   `load_reservations`, `load_clean_reservations`, `strip_pii` are), are **not** used
   in `main.py`, and `property_universe` is **never called anywhere** — grep shows
   only its definition plus its own internal call to `load_property_performance`.
3. **Its intended consumer is still a stub.** `dash_app/backend/locations.py::
   _load_locations()` hardcodes `return []` with the comment *"Deprecated placeholder
   until the SQL-backed property table is wired in."* So the path from the perf
   loader → the app's property/location universe was never completed.
4. **It fails soft, not loud.** `property_universe()` wraps the load in a broad
   `except Exception` and returns an **empty DataFrame** on any error. So even when
   "run," offline it silently yields an empty universe rather than an error.

The **table name is correct** (`property_performance_daily`) and the code itself is
well-formed — this is a "never connected + never called + consumer still stubbed"
situation, not a bug in the query. Wiring it up (build cache → export → feed the app)
is squarely a Step 2 task.

---

## 6. Discrepancies vs. the project instructions — please confirm before Step 2

Per the working agreement I'm stopping here rather than guessing on these:

1. **What is "model 8" for scoring? (biggest one.)** The brief says *"the current
   production model ('model 8')"* should be loaded and used to score reservations in
   Step 2. But in the code the **served/production model is the static XGBoost (02)**
   (`main.py` + `scoring.py` select the best static model; only `02` exists on disk).
   `08` is a **hazard** model with a *different output shape* (survival curve →
   cancel-by-arrival prob), served through a *separate* module, and *not* in the
   scoring registry. So:
   - **Which artifact should the Step 2 `score(reservations)` function use** — the
     static XGBoost `02` (what the code serves today), the hazard `08` (what "model
     8" literally names), or should it expose **both** (static for the booking-level
     risk table, hazard for per-night expected-freed-rooms)?
   - My recommendation, subject to your call: wrap **both** behind one clean
     interface — static `02` as the default per-booking probability, hazard `08`
     available for the time-to-arrival/per-night view the Occupancy dashboard needs.

2. **Missing model files.** The registry expects `01_logreg` and `03_histgb`, and the
   later XAI page expects a baseline/mean model — none are on disk. Fine for Phase 1
   (we only need one scoring model), but confirm you don't expect me to retrain them
   (the brief says do not retrain this phase).

3. **`property_performance_daily` is un-cached and its schema is unverified here.** I
   listed the intended columns from code (§3b) but could not confirm real dtypes,
   sample values, or the `propertyId → property_name` mapping without live BigQuery.
   Step 2 will need working creds (or a one-off cache you can hand me) to build the
   first `property_performance_daily.parquet` and to verify the join key between the
   two tables.

4. **Dependency gaps for the Dash foundation.** `pyproject.toml` has `dash`, `plotly`,
   `flask` but is **missing** `dash-ag-grid`, `dash-bootstrap-components` /
   `dash-mantine-components`, and a background-callback cache backend (`diskcache`,
   and later `celery`/`redis`). These will need adding in Step 2 (exact versions/APIs
   to be checked against each library's own docs, per the brief).

5. **Minor latent bug (non-blocking).** `main.py::cmd_status` computes
   `(df['status'] == 'Canceled').mean()` on the **clean** dataset, but there `status`
   is already encoded int8 (0/1) — so that line reports 0.00% instead of the true
   ≈19.8% base rate. Easy fix; noting it so it isn't mistaken for a data problem.

---

### Suggested Step 2 order (for reference — not started yet)

Foundation once §6.1 is answered: (a) multi-page Dash shell with `server =
app.server` and instant-loading placeholder pages; (b) data-access layer reusing the
existing loaders + building the property-performance Parquet cache + a manual
"refresh" trigger; (c) confirm ADC auth end-to-end; (d) a single `score(reservations)`
entry point over the chosen model(s); (e) add the missing Dash dependencies. No
charts yet.
