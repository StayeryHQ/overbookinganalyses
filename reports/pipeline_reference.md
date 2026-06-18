# Overbooking model pipeline — full reference

A teaching document for the model pipeline. Originally written for notebooks 01-04 (LogReg, RF, XGBoost, MLP); the lineup is now **LogReg (01) / XGBoost (02) / HistGB (03) + hazard (08) / XGB-AFT (09) / RSF (10)** - RF and MLP sections are kept as teaching material but those models are OUT (see reports/open_decisions.md). Two parts:

1. **Walkthrough** — read this top-to-bottom and you'll understand every line of the pipeline as it executes.
2. **Glossary** — A–Z lookup for every concept, parameter, and function. Use it when you forget what `handle_unknown` does.

Math is included where it actually teaches (L1 vs L2 penalty, sigmoid, log-loss, beta calibration). Skip the equations if they slow you down; the prose stands alone.

---

# PART 1 — WALKTHROUGH

## 0. The mental model — what is this pipeline even doing?

You have one row per hotel booking. Each booking either eventually got cancelled (`status = 1`) or didn't (`status = 0`). You want a function `f(booking_features) → P(cancel)` that gives you the probability of cancellation for any new booking, with two operational uses:

- **Ranking**: given a list of upcoming bookings, sort them by cancel risk. Top of the list = candidates for overbooking. Pure AUC / lift question.
- **Thresholded decision**: above some probability `p*`, flag the booking as "high risk". Below, ignore. This requires the probability to be *calibrated* — when the model says 0.78, the long-run cancel rate of bookings scored at 0.78 should actually be 78%.

The four model notebooks (01–04) all build the same skeleton:

```
Pipeline:
    prep:  ColumnTransformer
        ├── num     : SimpleImputer → StandardScaler           # numeric features
        ├── cat_lo  : SimpleImputer → OneHotEncoder            # low-card categoricals
        └── cat_hi  : SimpleImputer → RareCategoryCollapser → OneHotEncoder   # high-card

    clf:   CalibratedClassifierCV(base_classifier, method="isotonic", cv=5)
```

Each notebook swaps in a different `base_classifier`. Everything else is shared. The reason to keep four notebooks rather than one is so you can compare model families on a fair fight (same data, same preprocessing, same CV folds, same calibration).

---

## 1. Loading the cleaned parquet

```python
df = load_clean_reservations()
df["is_cancelled"] = df["status"].astype(int)
```

`load_clean_reservations()` reads `Data/reservations_clean.parquet`, the file produced by notebook 00 (data audit). At this point:

- Every row is a fully-resolved booking — either `Canceled` or `CheckedOut`/`InHouse` (the negatives).
- `status` is already binary `int8` (encoded in 00 §4.5). `is_cancelled` is just an alias for clarity in downstream code.
- The frame has ~188k rows × 29 columns. Most are engineered features built in 00 §3.0.
- `is_temporal_test` ∈ {0, 1} flags the newest 25% of rows by arrival. This is the test mask.

**Why a separate file?** To keep one source of truth for cleaning. If you change a cleaning rule in 00 and forget to update the model notebook, your scoring would diverge from training. By making the cleaned parquet the only contract, every model notebook *must* read it.

---

## 2. The feature roster

```python
> ⚠️ **LEAKAGE WARNING (2026-06-12):** `travelPurpose`, `primaryGuest_preferredLanguage`
> and `primaryGuest_address_countryCode` (-> `guest_country_region`) are completed at
> check-in - their missingness encodes the outcome (+0.10-0.12 offline AUC inflation,
> 82-88% predicted cancel prob on the 38% of upcoming bookings with empty profiles).
> They were REMOVED from all model rosters on 2026-06-12 (src/features.py NON_FEATURE_COLS); re-admission only via booking-time snapshots. Also:
> `is_international` was dropped in 00 §3.7; free-cancel features removed 2026-06-11.
> Evidence: experiments/profile_leakage_quantification.ipynb.

NUMERIC_FEATURES = [
    "lead_time_days", "los_nights", "adults_n",
    "log_gross_amount", "gross_per_night", "diff_gross_cancellation_fee",
    "arrival_dow", "arrival_month", "is_weekend_arrival",
    "has_promo", "has_corporate_code", "has_group",
]
CATEGORICAL_FEATURES = [
    "property_name", "channelCode", "ratePlan_category", "unitGroup_name",
    "guaranteeType", "stay_bucket", "cancellationFee_name",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_cancelled"
```

This is the only place in the notebook you choose which columns the model sees. The lists are filtered with `[c for c in [...] if c in df.columns]` so if a column gets dropped in 00 the model doesn't crash here — it just runs without that feature.

**Decision logged here.** OHE for low-cardinality (≤ ~30 levels). Rare-collapse + OHE for `primaryGuest_address_countryCode` (182 levels). `stay_bucket` is currently OHE — flagged in the open-decisions doc as something to switch to OrdinalEncoder later because it has a natural order (short < mid < long).

---

## 3. Train/test split — and why it's temporal, not random

```python
test_mask = df["is_temporal_test"].astype(bool)
X_train, X_test = X.loc[~test_mask].copy(), X.loc[test_mask].copy()
y_train, y_test = y.loc[~test_mask].copy(), y.loc[test_mask].copy()
```

**Random split fails for time-series data**. A `train_test_split(stratify=y, random_state=42)` puts bookings arriving in 2024 *and* 2025 into both train and test. The model is then asked "given 2024-Q3 and 2025-Q2 rows in training, predict a 2024-Q4 row in test". That's not how production works. In production, the model is *only* trained on past data and predicts future bookings.

A temporal split (train = arrivals before some cutoff, test = arrivals after) catches three problems random splits hide:

1. **Concept drift.** Channel mix shifts (more OTA, less direct over time). Pricing changes. Policy changes (e.g., adding non-refundable plans). The model trained yesterday slowly becomes wrong.
2. **Trend leakage.** If cancellation rates are rising over time, a random split silently lets the test rows "know" their position in time via correlated features. AUC looks better than it deserves.
3. **Calendar contamination.** Booking made 90 days ahead has 90 days of co-evolving market signal. Random splits scatter that co-evolution across folds.

In our pipeline:

```python
# Diagnostic — random split kept in parallel to see how badly the temporal
# split is hurting us (or how much the random split was lying).
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y,
)
```

You'll see two AUCs in the headline table — `test (temporal)` and `test (random, diag)`. The delta tells you how much temporal leakage your model was getting. If random > temporal by > 0.02 AUC, you have meaningful concept drift; if < 0.005, the static framing is holding up.

---

## 4. The `Pipeline` object — why we wrap everything

```python
pipeline = Pipeline([
    ("prep", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
    ("clf",  CalibratedClassifierCV(base_clf, method="isotonic", cv=5)),
])
```

A `Pipeline` is just a list of `(name, transformer)` tuples that get applied in sequence. The last one must implement `.predict()` / `.predict_proba()`; everything else just transforms data.

**Why this exists.** Each fit-transform step has to be re-fit on every training fold during CV. If you scaled by hand before the CV loop:

```python
# WRONG — leakage
scaler = StandardScaler().fit(X)    # the scaler's mean/std knows about validation rows
for train_idx, val_idx in cv.split(X):
    fit_model(X[train_idx]); evaluate(X[val_idx])
```

The scaler's mean is computed on the *entire* dataset including validation. Tiny leakage, but it's real. Same with the imputer, the OHE, the calibration wrapper. The Pipeline forces every component to be fit on the training fold only.

You can also access individual steps:

```python
pipeline.named_steps["prep"]       # the ColumnTransformer
pipeline.named_steps["clf"]        # the CalibratedClassifierCV
```

Useful for XAI in §9 of each notebook, where SHAP needs the inner base model.

---

## 5. `ColumnTransformer` — sending different columns through different transforms

```python
def build_preprocessor(numeric, categorical):
    hi_card = [c for c in categorical if c in HIGH_CARD_CATEGORICALS]
    lo_card = [c for c in categorical if c not in HIGH_CARD_CATEGORICALS]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler",  StandardScaler())])
    lo_pipe  = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))])
    hi_pipe  = Pipeline([("imputer",   SimpleImputer(strategy="most_frequent")),
                         ("collapse",  RareCategoryCollapser(min_count=10)),
                         ("ohe",       OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))])

    return ColumnTransformer(
        [("num", num_pipe, numeric),
         ("cat_lo", lo_pipe, lo_card),
         ("cat_hi", hi_pipe, hi_card)],
        verbose_feature_names_out=False,
    )
```

Three sub-pipelines, three different column families. The `ColumnTransformer` is the dispatcher: it slices the input DataFrame by column, sends each slice through the appropriate sub-pipeline, and `hstack`s the outputs.

### Parameters worth knowing

- **`remainder='drop'`** (default). Columns not listed in any transformer are discarded. The alternative `'passthrough'` keeps them untouched. We want `drop` so accidental columns (`arrival`, `created`, etc.) don't sneak into the model.
- **`sparse_threshold=0.3`** (default). If the combined output has < 30% non-zero entries, return a scipy sparse matrix; otherwise dense numpy. We're dense (most cells are 0/1 OHE but there are enough numeric columns). Matters when you have hundreds of thousands of OHE columns.
- **`verbose_feature_names_out=False`**. Output columns are named `channelCode_Direct` instead of `cat_lo__channelCode_Direct`. Just a quality-of-life setting for SHAP plots.

---

## 6. Numeric sub-pipeline

### 6.1 `SimpleImputer(strategy="median")`

```python
SimpleImputer(strategy="median")
```

Computes the median of each numeric column during `fit`. During `transform`, replaces every NaN in that column with the stored median.

**Why median, not mean?** `lead_time_days`, `gross_amount`, `los_nights` all have heavy right tails. One booking with a typo `gross_amount = €50,000` would drag the mean upward; the median doesn't move. Robust.

**Math.** Median = the value separating the higher and lower halves of the sorted column. For `[10, 20, 30, 40, 50,000]` the median is 30, the mean is 10,020.

**Alternatives.**

- `strategy="mean"` — fine for symmetric features.
- `strategy="constant", fill_value=0` — encodes "missing means zero".
- `IterativeImputer` — models each missing column as a function of the others. Slow.
- `KNNImputer` — uses the k nearest non-missing rows.

**State of the art** for tabular data is to *not impute* and use a model that handles NaN natively (XGBoost, LightGBM, CatBoost, HistGradientBoosting). They treat NaN as a third branch direction at each tree split, which often learns "NaN means something". Linear models still need imputation.

### 6.2 `StandardScaler`

```python
StandardScaler()
```

Subtracts the training-set mean, divides by the training-set standard deviation. Result: every numeric column has mean 0 and unit variance.

**Math.** For column `xⱼ`: `z = (x - μⱼ) / σⱼ` where `μⱼ, σⱼ` are estimated on training data only.

**Why scale?** Logistic regression's L2 penalty is `λ · Σ wⱼ²`. If `lead_time_days` ranges over [0, 365] and `is_weekend_arrival` over [0, 1], the penalty punishes `lead_time_days`'s coefficient ~365× more for the same information content. Scaling makes the penalty fair. Same logic for gradient-descent NNs.

**Trees don't need it.** A tree split asks `lead_time_days ≤ 30?` — only the order matters, not the magnitude. We keep the scaler in their pipelines anyway for consistency.

**Alternatives.**

- `MinMaxScaler` — maps to [0, 1]. Bad for outliers (one outlier compresses everyone else into a tiny range).
- `RobustScaler` — uses median and IQR instead of mean/std. Better when outliers can't be dropped.
- `PowerTransformer(method="yeo-johnson")` — makes the distribution Gaussian-shaped. Worth trying for mediocre logreg.
- `QuantileTransformer(output_distribution="normal")` — maps empirical CDF to Gaussian. Loses tail info.

**For NNs**: `LayerNorm` / `BatchNorm` inside the network. For our MLPClassifier we still use `StandardScaler` because sklearn's MLP doesn't include batch normalization.

---

## 7. Categorical sub-pipeline (low-cardinality)

These are: `property_name` (11), `channelCode` (13), `ratePlan_category` (59), `unitGroup_name` (10), `travelPurpose` (2), `guaranteeType` (3), `primaryGuest_preferredLanguage` (17), `stay_bucket` (3), `cancellationFee_name` (24).

### 7.1 `SimpleImputer(strategy="most_frequent")`

Replaces NaN with the most common value (mode) seen in the training column. For `travelPurpose` that would be "Leisure".

**When to switch.** If you suspect missingness itself is signal — e.g. a guest who didn't fill in `travelPurpose` cancels at a different rate than one who did — switch to `strategy="constant", fill_value="MISSING"`. The OHE will then learn a `MISSING` dummy column.

### 7.2 `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

For each categorical column with K levels, expand into K binary columns. `channelCode = "Direct"` becomes `channelCode_Direct = 1` and `channelCode_Ibe = 0`, etc.

**Math.** No equations. It's a lookup: each row contributes a 1 in exactly one of the K columns.

**Why this works for linear models.** Each level gets its own coefficient. The model learns "Direct bookings: coefficient −0.4 on cancel log-odds; Booking.com: coefficient +0.6". Interpretable.

**Parameters.**

- `handle_unknown="ignore"`. If a category at scoring time was never seen at training, encode it as all-zero across the K columns instead of raising. Defensive — daily scoring shouldn't crash on a new market.
- `sparse_output=False`. Return dense numpy. With small K this is fine and downstream code is cleaner.
- `drop=None` (default). Keeps all K columns. Setting `drop="first"` removes one column per feature to avoid multicollinearity — useful for unregularized linear models, irrelevant for our regularized ones.

**Alternatives** (this is where the field has moved on).

| Method | When to use | Cost |
|---|---|---|
| **OneHotEncoder** (current) | Low cardinality, linear models, interpretability | Many sparse columns |
| **OrdinalEncoder** | When the order matters: `stay_bucket` short < mid < long | Imposes false order if categories are nominal |
| **TargetEncoder** (sklearn 1.3+) | High cardinality + trees | Risks leakage if naive; sklearn handles via OOF |
| **category_encoders** library | 15+ encoders (leave-one-out, James-Stein, CatBoost-style) | Extra dependency |
| **CatBoost native** | If you switch classifier to CatBoost | No encoding needed at all |
| **Entity embeddings** | Very high cardinality + NN | Needs a NN architecture supporting it |

**State of the art (2024-25).**

- *Trees*: CatBoost's built-in categorical handling beats every external encoder on most benchmarks.
- *Linear models*: OHE still standard. Target encoding can help but you lose the per-level coefficient interpretability.
- *NNs on tabular*: entity embeddings (one learned vector per category, à la TabNet / SAINT / FT-Transformer).

---

## 8. Categorical sub-pipeline (high-cardinality)

Currently only `primaryGuest_address_countryCode` (182 unique).

### 8.1 `RareCategoryCollapser(min_count=10)` (custom)

```python
class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=10, other_label="Other"):
        self.min_count = min_count
        self.other_label = other_label
    def fit(self, X, y=None):
        s = pd.Series(X.iloc[:, 0]).astype("string")
        counts = s.value_counts()
        self.keep_ = set(counts[counts >= self.min_count].index)
        return self
    def transform(self, X):
        s = pd.Series(X.iloc[:, 0]).astype("string")
        return s.where(s.isin(self.keep_), other=self.other_label).to_numpy().reshape(-1, 1)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else self.feature_names_in_
```

Drops categories below the count threshold into "Other". With 182 countries, this typically yields ~20–35 "kept" plus "Other" — orders of magnitude fewer OHE columns.

**Why we built it.** Raw OHE on 182 countries gives 182 sparse columns, most firing in <10 rows. That's noise that wastes regularisation budget in linear models and offers zero signal to trees.

**Why `min_count` and not `top-k`.** A `min_count` rule is signal-driven: kept countries are those with enough data to estimate their effect. `top-k` is arbitrary — top-5 throws out countries 6-30 which had 50+ bookings each. Pick `min_count` for thresholds, `top-k` only if you need a hard cap on dummy column count.

**The Liechtenstein case.** A new country at scoring time (`"LI"`) is not in `self.keep_` (never seen in training), so the `.where` clause maps it to `"Other"`. The OHE then sees `"Other"`, which *is* a known training-time category (because rare countries were folded into it). So `countryCode_Other = 1` for that booking. Works automatically.

**`get_feature_names_out`.** This method is required for sklearn's downstream introspection (used by SHAP for feature naming). Since the collapser is 1-column-in / 1-column-out, it just passes the input name through.

### 8.2 OneHotEncoder (same as 7.2)

Encodes the post-collapse column. Now `K` is ~25 instead of 182.

### 8.3 Alternatives for `countryCode` specifically

- **`TargetEncoder`** (sklearn 1.3+). Replaces "DE" with the smoothed mean cancel rate of Germans in training. One column instead of 25. Bigger signal density. Risk: leakage if naively implemented; sklearn's handles it via internal OOF.
- **Frequency encoding**. Replace "DE" with the count or share of "DE" in training. Cheap but lossy.
- **Hashing trick** (`HashingEncoder`). Hashes each level into one of N buckets. For 5000+ levels; overkill at 182.

Per the open-decisions doc, `TargetEncoder` is task-tracked for later — typically wins +0.005-0.015 AUC over collapse+OHE.

---

## 9. The four classifiers

Each notebook (01–04) plugs a different `base_clf` into the same Pipeline skeleton. The classifiers are then wrapped in `CalibratedClassifierCV`. We'll cover the wrapper in §10.

### 9.1 LogisticRegression with ElasticNet (notebook 01)

```python
base_clf = LogisticRegression(
    penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga",
    max_iter=2000, random_state=RANDOM_STATE,
)
```

**The model.** Compute `z = w·x + b` (a linear combination of features), then `p = sigmoid(z) = 1 / (1 + e⁻ᶻ)`. `p` is the predicted probability of cancellation.

**Sigmoid graph (mental picture).**

```
p
1.0 |               _________
    |           __--
0.5 |        --
    |    __--
0.0 | __-
    |____________________________
     -6   -3    0    3    6     z
```

S-shaped. `z = 0 → p = 0.5`, `z = -∞ → p = 0`, `z = +∞ → p = 1`.

**Training.** Minimize log-loss (cross-entropy):

```
L = -Σᵢ [ yᵢ · log(pᵢ) + (1 - yᵢ) · log(1 - pᵢ) ] + λ · penalty(w)
```

The first part is log-loss; the second is regularisation. The optimiser (`saga`) finds the `w` that minimises this.

#### Why ElasticNet

Regularisation prevents over-fitting and feature-by-feature noise. Two flavours:

- **L2** (Ridge): `penalty = ½ Σ wⱼ²`. Penalises *squared* coefficients. Shrinks all toward zero but rarely to *exactly* zero.
- **L1** (Lasso): `penalty = Σ |wⱼ|`. Penalises *absolute* coefficients. Drives many to *exactly* zero — automatic feature selection.

ElasticNet mixes them:

```
penalty = α · Σ|wⱼ| + (1-α)/2 · Σwⱼ²
```

where `α = l1_ratio`. `l1_ratio=0.5` is the standard balanced default.

**Why mix?** With ~80 OHE dummies, many of which are noisy or correlated:

- L1 alone: picks one of a correlated pair arbitrarily (e.g. `ratePlan_corporate` vs `corporateCode`-derived flag). Unstable across CV folds.
- L2 alone: keeps every dummy, just shrinks them. No sparsity.
- ElasticNet: sparsity *and* stability. Industry default.

**Parameters.**

- `C=1.0`. Inverse regularisation strength. Higher C → less regularisation. The standard knob to tune via CV.
- `l1_ratio=0.5`. The mix knob. 0 = pure L2, 1 = pure L1.
- `solver="saga"`. The only sklearn solver supporting ElasticNet. It's a stochastic-gradient variant of SAG (Stochastic Average Gradient); slow but converges.
- `max_iter=2000`. Convergence ceiling. If you see `ConvergenceWarning`, bump it up.
- `random_state=RANDOM_STATE`. Reproducibility.

**Alternatives.**

- `penalty="l2"` (default). The safe choice; no sparsity.
- `penalty="l1"`. Maximum sparsity; unstable on correlated features.
- `penalty=None`. Unregularised — overfits on 80+ features.
- Solvers: `liblinear` (fast, L1/L2 only — no ElasticNet), `lbfgs` (fast, L2 only), `newton-cg` (L2 only, second-order), `sag` (large data, L2 only).

**State of the art** for linear classification is still ElasticNet. For very high-dimensional data: group-lasso, OWL, or specialised solvers (GLMNet, Vowpal Wabbit). For our scale, sklearn saga is fine.

### 9.2 RandomForestClassifier — ⚠️ MODEL OUT OF LINEUP (kept as teaching material)

```python
base_clf = RandomForestClassifier(
    n_estimators=400, max_depth=14, min_samples_leaf=20,
    max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
)
```

**The model.** Train 400 separate decision trees, each on a bootstrap sample of the training data (random sampling with replacement). At each split inside each tree, only `sqrt(n_features)` features are considered (so the trees disagree more). Final prediction = average of the trees' predicted probabilities.

**Why this works.** Single decision trees overfit dramatically. But many uncorrelated trees, *averaged*, smooth out the overfitting. This is "bagging" — Bootstrap AGGregating. The variance of the average is `Var(tree) / n_trees` (modulo correlation), so more trees = less variance.

**Parameters.**

- `n_estimators=400`. Number of trees. More is better but with diminishing returns past ~200-500. Cost: linear.
- `max_depth=14`. Maximum tree depth. The previous default `None` (grow until pure) produced trees with thousands of leaves, slow scoring and overconfident probabilities. 14 is a sensible cap on this data.
- `min_samples_leaf=20`. Each leaf must contain at least 20 training rows. Prevents trees from memorising individual rows. Higher = smoother predictions, less overfitting.
- `max_features="sqrt"`. At each split consider `sqrt(n_features) ≈ 9` candidates (out of ~80 features). Forces tree decorrelation. `"log2"` is similar; `1.0` means use all features (basically just bagging without feature subsampling).
- `n_jobs=-1`. Use all CPU cores.
- `random_state=RANDOM_STATE`. Reproducibility.

**Note**: we removed `class_weight="balanced_subsample"` from the original. That was distorting the probability scale (see calibration discussion in §10). We rely on calibration instead.

**Alternatives within the same family.**

- `HistGradientBoostingClassifier`. Faster than RF, calibrated by default, handles NaN natively. Worth swapping in if RF underperforms. Tracked as a task.
- `ExtraTreesClassifier`. Like RF but uses random split thresholds (not optimal ones). Faster, slightly worse AUC on most problems.

**State of the art** for "I want a tree ensemble that works": LightGBM > XGBoost > CatBoost > HistGB > RF, roughly, on most tabular tasks. CatBoost wins when categoricals are dominant.

### 9.3 XGBoost (notebook 03)

```python
base_clf = XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0,
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
)
```

**The model.** Gradient boosting. Start with a constant prediction (the base rate). Then iteratively add small trees: each new tree is fit to predict the *residuals* (or more precisely, the gradient of the loss) of the ensemble built so far. Final prediction = sum of all trees' contributions, passed through sigmoid.

**Why this works.** Each tree fixes the errors of the previous trees. Boosting is a sequential, deliberate process — unlike RF's "average many independent guesses". With shallow trees and small learning rates, you get an extremely flexible model that resists overfitting.

**Parameters.**

- `n_estimators=600`. Number of trees (boosting rounds). Combined with `learning_rate=0.05`: total "model capacity" = `n_estimators × learning_rate ≈ 30`. With early stopping on a validation set you'd let it run longer and stop automatically.
- `max_depth=6`. Trees are *shallow* (6 levels). Boosting works best with weak learners — each tree captures one specific kind of pattern.
- `learning_rate=0.05`. Step size. Smaller = more conservative; needs more trees but generalizes better.
- `subsample=0.8`. Each tree fits on 80% of training rows (random sample, no replacement). Bagging-style regularisation.
- `colsample_bytree=0.8`. Each tree sees 80% of features. Decorrelates trees.
- `reg_lambda=1.0`. L2 regularisation on leaf values.
- `reg_alpha=0.0`. L1 regularisation on leaf values. Set to >0 for sparser models.
- `objective="binary:logistic"`. Loss function = log-loss; output = sigmoid.
- `eval_metric="auc"`. Metric tracked during training (used by early stopping if enabled).
- `tree_method="hist"`. Use histogram-based split finding (like LightGBM). 5-10× faster than the exact method.

**Note**: removed `scale_pos_weight = neg/pos` from the original. Same reason as RF — it distorts the probability scale.

**Alternatives.**

- LightGBM — equivalent to XGB hist mode, slightly faster, slightly worse on small data.
- CatBoost — built-in categorical handling, ordered boosting (reduces target leakage), best for categorical-heavy data.
- HistGradientBoostingClassifier — sklearn's native equivalent. Faster than XGBoost API, fewer features, no GPU.

**State of the art** for tabular ML in 2024-25 is "XGBoost or LightGBM or CatBoost — pick one and tune it". Differences between the three are usually < 0.005 AUC after tuning.

### 9.4 MLPClassifier — ⚠️ MODEL OUT OF LINEUP (kept as teaching material)

```python
base_clf = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu",
    alpha=1e-3, learning_rate_init=1e-3, solver="adam",
    max_iter=200, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=10,
    batch_size="auto", random_state=RANDOM_STATE,
)
```

**The model.** A simple feed-forward neural network. Input features → 64-neuron hidden layer (ReLU activation) → 32-neuron hidden layer (ReLU) → 1-neuron output (sigmoid). Each layer = matrix multiply + nonlinearity.

**Math intuition.** A two-hidden-layer MLP with ReLU is a *universal function approximator* — given enough neurons, it can approximate any continuous function. With (64, 32) we're constrained but expressive.

**Parameters.**

- `hidden_layer_sizes=(64, 32)`. Two hidden layers, 64 and 32 neurons.
- `activation="relu"`. Rectified Linear Unit: `f(x) = max(0, x)`. Default for hidden layers; cheap and works.
- `alpha=1e-3`. L2 weight regularisation strength. Larger = simpler model.
- `learning_rate_init=1e-3`. Adam optimiser starting step size.
- `solver="adam"`. Adaptive gradient method. Stochastic, robust, default for small NNs.
- `max_iter=200`. Maximum training epochs.
- `early_stopping=True`. Hold out 10% of training data; stop when validation loss hasn't improved for `n_iter_no_change=10` epochs.
- `validation_fraction=0.1`. Internal validation hold-out fraction.
- `batch_size="auto"` = min(200, n_samples). Mini-batch size for stochastic updates.

**Why we have it.** Sanity floor against the tree models. Tabular MLPs at this size rarely beat XGBoost. If it does, that's a discovery worth investigating.

**Alternatives.**

- Bigger MLP (256, 128, 64) + dropout. Often helps but risks overfit.
- TabNet, FT-Transformer, SAINT — purpose-built tabular NNs. State of the art for tabular DL but complicated.
- Trees almost always win on this problem family. The case where NNs win: very high-cardinality categoricals with strong interactions (entity embeddings).

---

## 10. CalibratedClassifierCV — the wrapper

```python
calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
pipeline = Pipeline([("prep", prep), ("clf", calibrated_clf)])
```

The most subtle component in your pipeline. It does *not* train a new model; it learns a rescaling that maps the base model's raw output into well-calibrated probabilities.

### 10.1 Why this is needed — softmax/sigmoid is not calibration

A common misconception: "the output of softmax/sigmoid is already between 0 and 1, so it's a probability". It is a number in [0, 1] but not necessarily a *frequentist* probability — meaning, the long-run rate of positives among examples scored at that value.

**Concrete example.** Your XGBoost says "0.83 cancel" for 1000 bookings. For the model to be calibrated, ~830 of those 1000 should actually cancel. Without calibration, often only 650 do (overconfident) or 920 do (underconfident).

**Why training pushes scores toward miscalibration.** Cross-entropy loss `-y log(p) - (1-y) log(1-p)`:

- True positive scored at p=0.99 → loss ≈ 0.01 (small gradient, model is satisfied)
- True positive scored at p=0.51 → loss ≈ 0.67 (large gradient, model gets pushed)

So gradient descent keeps pushing high-confidence predictions further out toward 0 or 1 even after the rank ordering is already correct. AUC stays stable; calibration drifts.

**Per-model miscalibration patterns.**

| Model | Typical pattern | Why |
|---|---|---|
| LogReg (L2) | Mild under-confidence | L2 shrinks coefficients, sigmoid pulled toward 0.5 |
| RF | **U-shaped** | Each tree votes hard; averaging accumulates at extremes |
| XGBoost | Overconfident at extremes | Many trees, large logits, sigmoid saturates |
| MLP | Severely overconfident | Modern NN training is the worst offender |
| Naive Bayes | Severely overconfident | Independence assumption inflates likelihoods |

### 10.2 What `CalibratedClassifierCV` actually does

With `cv=5, method="isotonic"`:

1. Split training data into 5 inner folds.
2. For each fold, train the base classifier on the other 4 folds.
3. Predict raw probabilities on the held-out fold (giving OOF raw scores for every training row).
4. Collect all (raw_score, true_label) pairs across folds.
5. Fit an isotonic regression mapping raw_score → calibrated_probability.
6. At prediction time: each of the 5 base models is applied, their outputs are averaged (forming the raw score), then passed through the isotonic map.

This means **one fit of `CalibratedClassifierCV` does 5 base-model fits**. That's the 5× multiplier that made our CV cell so slow before optimisation — see §11.

### 10.3 Method choices

#### `method="isotonic"`

Fits a monotone non-decreasing step function. Non-parametric. Can fit any monotone miscalibration curve.

```
calibrated p
    1.0 |                         ______
        |                    ____|
    0.7 |             ______|
        |        ____|
    0.4 |   ____|
        |  |
    0.0 +-----------------------------------> raw score
        0.0       0.5       1.0
```

**Strengths.** Flexible. No hyperparameters. Works for any shape.

**Weaknesses.** Step function = jumpy. Needs > ~500-1000 calibration rows per fold to fit reliable bins. With our ~24k per fold, we're fine.

#### `method="sigmoid"` (Platt scaling)

Fits a 2-parameter logistic: `calibrated_p = sigmoid(a · raw + b)`. Smooth, no overfitting, works on small data. But assumes the miscalibration is sigmoid-shaped — fails on U-shaped miscalibration (RF).

#### Beta calibration (Kull et al. 2017) — not in sklearn

`calibrated_p = sigmoid(a · log(p) + b · log(1-p) + c)`. Three parameters. Smooth like Platt but more flexible (Platt is a special case where `a = -b`). Theoretically optimal when raw scores are approximately Beta-distributed (which they often are).

To use it: `pip install betacal`, then `BetaCalibration(parameters="abm").fit(oof, y_true)`.

**Choice for our project.** With 24k rows per CV fold, isotonic is reasonable. Beta would be marginally smoother but requires installing a package. The actual answer comes from the reliability diagram in §7 of each notebook — if it looks jaggy, swap to beta.

### 10.4 Parameters

- `base_estimator` (positional). The model to calibrate.
- `method`. `"isotonic"` or `"sigmoid"`.
- `cv=5`. Inner CV folds. More = better calibration map, slower.
- `n_jobs`. Parallelism. Watch out for nested parallelism — if you set both this and your base model's `n_jobs=-1`, you can deadlock or thrash.

---

## 11. Cross-validation

```python
# The speed trick: strip the calibration layer for CV
cv_pipeline = Pipeline([
    ("prep", clone(pipeline.named_steps["prep"])),
    ("clf",  clone(pipeline.named_steps["clf"].estimator)),
])

tss = TimeSeriesSplit(n_splits=3)
res_t = cross_validate(cv_pipeline, X_train_o, y_train_o, cv=tss,
                       scoring=list(SCORING), return_train_score=False, n_jobs=-1)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
res_s = cross_validate(cv_pipeline, X_train, y_train, cv=skf,
                       scoring=list(SCORING), return_train_score=False, n_jobs=-1)
```

### 11.1 Why CV at all

A single train/test split gives one number; it could be lucky or unlucky. Cross-validation splits the training data into k folds, trains on k-1 folds, evaluates on the held-out fold, rotates. You get k metric values whose mean estimates true generalization and whose std estimates how much the model's performance fluctuates from one slice of data to another.

### 11.2 `StratifiedKFold(shuffle=True, n_splits=3)` — random folds

Splits randomly with stratification (each fold has roughly the same positive class share as the whole training set). Standard for classification on i.i.d. data.

**Problem for time-series.** Random shuffling leaks the future into the past — fold 1 might be from 2024-Q4, fold 2 from 2023-Q1, etc. The model exploits temporal information at training time that wouldn't be available at deployment.

We keep it as a *diagnostic*. If random-CV AUC is much higher than time-CV AUC, that's evidence of temporal drift.

### 11.3 `TimeSeriesSplit(n_splits=3)` — chronological folds

```
data sorted by arrival ────────────────────────────────────────►
fold 1   train [0..25%]   val [25..50%]
fold 2   train [0..50%]   val [50..75%]
fold 3   train [0..75%]   val [75..100%]
```

Each fold trains on the past, validates on the next chunk of the future. Mimics deployment: model is built on all data up to date X, applied after date X.

**The asymmetry that bites.** Early rows are in *training* of every fold. Late rows are in *validation* of one fold. Folds aren't a partition. This breaks `cross_val_predict` — see §11.6.

### 11.4 `cross_validate`

```python
cross_validate(estimator, X, y, cv=cv, scoring=list_of_metrics, n_jobs=-1)
```

Runs CV. For each fold: fit on the train portion, score on the validation portion. Returns a dict with arrays of length `n_splits`:

- `test_<metric>` — validation scores per fold.
- `train_<metric>` — training scores (only if `return_train_score=True`).
- `fit_time`, `score_time` — durations per fold.

We pass `return_train_score=False` to save time (each `train_<metric>` requires re-scoring on the training portion, doubling scoring cost).

### 11.5 Scoring metrics

```python
SCORING = ("roc_auc", "average_precision", "f1", "neg_brier_score")
```

These are sklearn-recognised strings. `cross_validate` computes each independently. AUC and AP are computed from probabilities (`predict_proba`), F1 from labels (using default threshold 0.5), Brier from probabilities.

**`neg_brier_score`**: sklearn convention is "higher is better" for all metrics, but Brier is a *loss* (lower is better). The negative sign flips it so `cross_validate` can rank consistently.

### 11.6 `cross_val_predict` — and why it fails with TimeSeriesSplit

`cross_val_predict` returns one prediction per row. For that to be well-defined, every row must appear in *exactly one* validation fold (a *partition*). `KFold` and `StratifiedKFold` do this. `TimeSeriesSplit` doesn't — early rows are never in any val fold, late rows are in only one. sklearn raises `ValueError: cross_val_predict only works for partitions`.

Workaround (in cell 18):

```python
oof_prob = np.full(len(X_train_o), np.nan)
for train_idx, val_idx in tss.split(X_train_o):
    fold_pipe = clone(cv_pipeline)
    fold_pipe.fit(X_train_o.iloc[train_idx], y_train_o.iloc[train_idx])
    oof_prob[val_idx] = fold_pipe.predict_proba(X_train_o.iloc[val_idx])[:, 1]

mask = ~np.isnan(oof_prob)
y_tuning, prob_tuning = y_train_o.values[mask], oof_prob[mask]
```

Manual loop. Each row that was in *some* val fold gets a prediction; rows never in any val fold stay NaN and are masked out before threshold tuning.

### 11.7 The dual-CV strategy (and why it's so slow without the trick)

We run CV twice — TSS and SKF — to expose temporal drift.

**Before the optimisation**, each `cross_validate` did `5 outer folds × 5 inner calibration folds = 25 base fits`. With two CVs: 50 base fits total. With LogReg-saga or RF, that's 30+ minutes.

**The trick.** Calibration is a monotone rescaling — it doesn't change ranking, so AUC/AP/F1 (the metrics we use in CV) are identical with or without it. Strip the calibration from the CV pipeline:

```python
cv_pipeline = Pipeline([("prep", clone(prep)), ("clf", clone(base))])
```

Now each `cross_validate` does `3 outer folds × 1 base fit = 3 base fits`. Two CVs = 6 base fits total. ~8× speedup, zero methodological loss.

You only pay the calibration cost once: when you call `pipeline.fit(X_train, y_train)` in §5 of the notebook for the final fit + test prediction.

---

## 12. Threshold tuning

Step 1: a classifier's `predict_proba` gives `p ∈ [0, 1]`. To turn that into a "flagged / not flagged" decision you pick a threshold and apply `flag = (p ≥ threshold)`.

Step 2: the default `0.5` is rarely the best threshold operationally — base rate is 21% positive, so most bookings will be below 0.5 even if the model is good.

### 12.1 Three operating points (the cell at 18)

#### F1-optimal — balanced

```
F1 = 2 · precision · recall / (precision + recall)
```

Harmonic mean of precision and recall. Take the threshold where F1 is maximal on OOF training predictions. The math:

```python
prec, rec, thr = precision_recall_curve(y_oof, prob_oof)
f1 = 2 * prec * rec / (prec + rec)
best_idx = int(np.nanargmax(f1[:-1]))
thr_f1 = thr[best_idx]
```

#### Precision-first — conservative

```python
RECALL_FLOOR = 0.50
candidates = np.where(rec >= RECALL_FLOOR)[0]
best_idx_p = candidates[np.argmax(prec[candidates])]
thr_p = thr[best_idx_p]
```

Among thresholds achieving recall ≥ 50%, pick the one with highest precision. Use when false positives are expensive (over-flagging non-cancellations) — for our business, this is the typical case.

#### Recall-first — aggressive

```python
PRECISION_FLOOR = 0.50
candidates = np.where(prec >= PRECISION_FLOOR)[0]
best_idx_r = candidates[np.argmax(rec[candidates])]
thr_r = thr[best_idx_r]
```

Among thresholds achieving precision ≥ 50%, pick the one with highest recall. Use when missing a cancellation is expensive.

### 12.2 Why tune off the test set

The mistake (was in the original code): pick the threshold by argmax F1 on `(y_test, y_prob_test)`, then report precision/recall at that threshold *on the same y_test*. You used the test set to both pick the operating point AND evaluate it — the reported precision/recall are upward-biased.

The fix: pick the threshold on OOF training predictions; apply it once to the test set. The test set is touched exactly once for reporting.

### 12.3 The sweep table

Beyond the three named points, the cell also reports metrics at fixed thresholds `[0.10, 0.20, ..., 0.90]`. Columns:

- `threshold` — the cutoff
- `n_flagged` — how many test bookings exceeded it
- `pct_flagged` — same as fraction
- `precision` — `TP / (TP + FP)`
- `recall` — `TP / (TP + FN)`
- `f1` — harmonic mean
- `lift` — `precision / base_rate`. A lift of 3 means: bookings flagged at this threshold cancel 3× more often than a random booking. Directly maps to the dashboard's bucket thinking.

---

## 13. Test-set metrics

Computed on the temporal hold-out (`y_test`, `y_prob_test`).

### 13.1 ROC curve and AUC

ROC = "Receiver Operating Characteristic". Plot TPR (=recall) on Y vs FPR on X, sweeping the threshold from 1 to 0.

```
TPR = TP / (TP + FN)        # true positive rate, aka recall, aka sensitivity
FPR = FP / (FP + TN)        # false positive rate
```

**AUC** = area under that curve. Interpretation: probability that a randomly chosen positive has a higher score than a randomly chosen negative. 0.5 = random, 1.0 = perfect.

**Strength.** Threshold-free; measures ranking quality.

**Weakness.** On imbalanced data, AUC overstates utility — moving from FPR 0.01 to 0.02 looks like "1% of the curve" but doubles your false alarms.

### 13.2 PR curve and Average Precision (AP)

Plot precision vs recall, sweeping the threshold.

**AP** = average of precisions at each recall point. Equivalent to area under the PR curve.

**Why also report this.** Imbalanced data: positives are rare. PR curve focuses on them. AUC of 0.85 with 21% positives could still mean "lots of false positives at any usable threshold"; AP makes that visible.

### 13.3 F1 score (already covered in §12)

```
F1 = 2 · precision · recall / (precision + recall)
```

Computed at a specific threshold (default 0.5 in the headline; we report it at the OOF-tuned threshold in §6).

### 13.4 Brier score

```
Brier = (1/N) · Σᵢ (pᵢ - yᵢ)²
```

Mean squared error between probabilities and true labels. A *proper scoring rule* — it's minimised only when the model outputs true probabilities. Lower is better. Perfect = 0. Always predicting the base rate (e.g. 0.21) gives Brier = base_rate · (1 - base_rate) ≈ 0.166 in our case.

**Why include it.** Brier penalises miscalibration. If two models have the same AUC but model A outputs (0.99, 0.01) for confident predictions and model B outputs (0.85, 0.15), and the *true* rates among those are also (0.85, 0.15) — then model A has worse Brier, model B is better calibrated.

### 13.5 Confusion matrix

For a chosen threshold, classify every test row as flagged (1) or not (0); cross with the true label.

```
                  predicted 0      predicted 1
true 0   |   TN              |   FP              |
true 1   |   FN              |   TP              |
```

The 1×3 panel in cell 18 shows confusion matrices at all three named thresholds side-by-side. Read it as: how many TPs vs FPs vs FNs vs TNs do you live with at this operating point?

### 13.6 Lift@k

```
lift@k = precision_at_top_k / base_rate
```

Sort all test predictions by score, take the top k%, compute precision in that slice, divide by the population base rate. Operational interpretation: how much better than random is the model at picking out cancellations?

- `lift@5%` = precision in top 5% / 21% base rate
- `lift@10%`, `lift@25%` similarly

If you can review the top 10% of bookings flagged, you want `lift@10%` to be high. A lift of 3.0 means flagged bookings cancel 3× more often than the average booking.

### 13.7 Precision at fixed recall

`prec@recall=0.6`: among all thresholds achieving recall ≥ 60%, what's the highest precision? Directly answers "if I have to catch 60% of cancellations, what's the false-flag rate I have to tolerate?"

---

## 14. Reliability diagram (the calibration check)

```python
prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=12, strategy="quantile")
ax.plot([0, 1], [0, 1], "--", label="perfect")
ax.plot(prob_pred, prob_true, "-o", label="model")
```

Bin the test predictions by score; within each bin compute the mean predicted probability and the mean true label. Plot one against the other.

**Perfect calibration** = points on the diagonal.

**Above diagonal** at high scores: model said 0.9, reality was 0.95 → *under-confident*.

**Below diagonal** at high scores: model said 0.9, reality was 0.75 → *over-confident*.

`strategy="quantile"` makes each bin contain the same number of test rows. `strategy="uniform"` (default) divides [0,1] into 12 equal-width bins; bins near 0.5 will be empty for imbalanced data.

**What we expect post-calibration.** The reliability curve should hug the diagonal. If it doesn't, isotonic calibration isn't doing its job — either you didn't have enough OOF data per fold, or the miscalibration was severely U-shaped at a place isotonic couldn't bin enough samples.

---

## 15. Per-slice performance

Same metrics, but computed within population subsets.

### 15.1 Per-property (kept — overbookings happen per property)

```python
auc_by_slice("property_name", labels=X_te["property_name"], y_true, y_prob, min_rows=30)
```

For each property, compute AUC on its test rows. If a property has <30 rows or all-same labels (so AUC is undefined), skip it.

**Why it matters.** Overbooking decisions are made per property. A model with 0.85 overall AUC that's 0.60 on one property is not deployable for that property; the dashboard would mislead the team running it.

### 15.2 Per channel

`Direct` (cancel rates near 5-10%) vs `Booking.com` (often 25-30%) vs corporate channels (often very low) — completely different distributions. If your model is great on OTA but mediocre on Direct, you need to know.

### 15.3 Per ratePlan_category

Non-refundable plans have ~5% cancel rate; flexible plans have 30-40%. The model should crush AUC on non-refundable (because there's a strong structural prior) and struggle relatively on flexible (where the signal is genuinely noisier).

### 15.4 Per lead-time bucket

```python
bins   = [-1, 7, 30, 90, 365]
labels = ["0-7d", "8-30d", "31-90d", "90+d"]
```

Different lead times have different cancellation dynamics. The model's AUC at d=0-7d is operationally most important (this is when overbooking decisions are actually made). If AUC drops sharply at small d, that's the signal to look at the hazard model in 08.

---

## 16. Explainable AI (lecture-style)

Three families: model-agnostic permutation, partial dependence + ICE, and SHAP. All three appear in `resources/p6_xai.ipynb` and we mirror that style.

### 16.1 Permutation importance (bar + box)

```python
perm = permutation_importance(pipeline, X_test, y_test, n_repeats=5,
                              scoring="roc_auc", random_state=RANDOM_STATE)
```

For each feature: randomly shuffle that one column in X_test (breaking its relationship with y), recompute test AUC, observe the drop. Repeat 5 times to get a distribution.

The *mean drop* is the feature's importance. The *distribution across repeats* is shown as a box plot — a feature whose mean is high but whose box is wide is *unstable*; treat its rank with suspicion.

**Why model-agnostic.** Works for any model — only needs `score()` access. Computationally expensive (one full prediction pass per feature per repeat).

**Caveats.**

- Penalises correlated features less than impurity-based importance, but still imperfect. If `lead_time_days` and `is_within_7d_of_arrival` are highly correlated, shuffling one of them still leaves the model some signal via the other.
- Computed on test data — measures importance for *generalization*, not for fitting training data. Right metric for "what does this model use to predict on new data?"

### 16.2 Partial Dependence Plot (PDP) + ICE overlay

```python
PartialDependenceDisplay.from_estimator(
    pipeline, X_train.sample(1000), features=["lead_time_days"],
    kind="both", ...
)
```

**PDP.** Pick a feature `xⱼ`. For each value `v` in a grid over the feature's range: take all training rows, set their `xⱼ = v`, predict, average. Plot the curve.

Reads as: "*holding everything else fixed (at its observed distribution), how does the prediction change as `xⱼ` varies?*"

**ICE.** Same sweep, but instead of averaging keep one curve per row. Reveals heterogeneity — maybe the effect of `lead_time_days` is monotone-decreasing for Direct bookings but flat for OTA. The PDP would average these out and look misleading.

**Caveat.** PDP / ICE assume features are independent. If `lead_time_days` and `arrival_month` are correlated (people booking in December often book 3 weeks out), the PDP sweep generates rows like `lead_time_days=180, arrival_month=December` that don't actually exist in reality. SHAP dependence (next) handles this better.

### 16.3 SHAP

SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature contributions that sum to (prediction − base rate). It's game-theoretic: each feature gets credit equal to its average marginal contribution across all possible subsets of other features. Mathematically principled.

Five plots, four shown in our notebook:

#### Waterfall (local)

```python
shap.plots.waterfall(shap_values[i])
```

Explains a single prediction. Bars show each feature's contribution. Starts from `E[f(X)]` (base rate) and ends at `f(X_i)`.

#### Beeswarm (global)

```python
shap.plots.beeswarm(shap_values, max_display=15)
```

For each feature (rows): a row of dots, each dot = one test row. X-axis = SHAP value (how much this feature pushed prediction up or down). Color = feature value (high or low). Reveals direction of effect AND heterogeneity AND interactions.

This is the single most informative XAI plot in the notebook. Read it left-to-right and you'll see:

- which features matter (long horizontal spread = high importance);
- in which direction (left of zero = pushes toward "not cancel", right = pushes toward "cancel");
- whether high or low values of the feature drive it (color);
- and how heterogeneous the effect is (vertical thickness).

#### Bar (global mean |SHAP|)

```python
shap.plots.bar(shap_values, max_display=15)
```

For each feature, the mean of `|SHAP|`. A faster, principled alternative to permutation importance for tree models.

#### Scatter dependence

```python
shap.plots.scatter(shap_values[:, "lead_time_days"], color=shap_values)
```

Like a PDP but each dot is a real row. X = feature value; Y = SHAP value for that row. Color = a second feature (auto-picked by SHAP for strongest interaction). Reveals nonlinearity and interactions in one plot.

#### Force plot (not in our notebook)

`shap.plots.force(shap_values[i])` — same as waterfall but horizontal and condensed. Stylistic choice. In the lecture (p6_xai) both waterfall and force are used.

### 16.4 SHAP choice of explainer

```python
if MODEL_LABEL in {"RF", "XGB"}:
    explainer = shap.TreeExplainer(base_model)
else:
    explainer = shap.KernelExplainer(...)
```

- **`TreeExplainer`** — exact SHAP values for tree models in polynomial time. Fast. Use for RF, XGB, LightGBM, HistGB, CatBoost.
- **`KernelExplainer`** — model-agnostic, approximates SHAP via local linear models. Slow (O(2^features) in the worst case, sampled to a reasonable bound). Use for LogReg, MLP, anything where TreeExplainer doesn't apply.
- **`DeepExplainer`** — for deep NNs in TF/PyTorch. Faster than Kernel for NNs.
- **`LinearExplainer`** — for linear models. SHAP values reduce to standardised coefficients × standardised inputs. Fast.

For our LogReg (notebook 01), `LinearExplainer` would be the right choice. We use Kernel for simplicity / consistency across notebooks.

---

# PART 2 — GLOSSARY (A–Z)

Lookup reference. Each entry: definition, when used, code example, alternatives where applicable.

## A

### `accuracy_score(y_true, y_pred)`

Fraction of predictions equal to true labels: `(TP + TN) / N`. **Misleading on imbalanced data** — predicting the majority class always gets ~80% accuracy on a 20%-positive problem. We compute it but don't use it for model selection.

### Adam (optimiser)

Adaptive Moment Estimation. Stochastic gradient descent with adaptive per-parameter learning rates. Default optimiser for neural networks; works well out-of-the-box. Used by `MLPClassifier(solver="adam")`.

### AP / Average Precision

Area under the precision-recall curve. `average_precision_score(y_true, y_prob)`. Better than AUC for imbalanced classification. Higher is better; perfect = 1.0; random = base_rate.

### AUC / ROC-AUC

Area under the ROC curve. Probability that a randomly drawn positive scores higher than a randomly drawn negative. Threshold-free ranking quality metric. 0.5 = random, 1.0 = perfect. `roc_auc_score(y_true, y_prob)`.

## B

### Bagging (Bootstrap AGGregating)

Train many models on bootstrap samples of the data, average their predictions. Used by RandomForest. Reduces variance, doesn't reduce bias. Good for high-variance base learners (deep trees).

### Base rate

The proportion of positives in the dataset. Our base rate is ~21%. Used in lift calculations (`lift = precision / base_rate`) and as the reference for "no information" predictions.

### Beta calibration

A calibration method (Kull et al. 2017): `p_calibrated = sigmoid(a · log(p) + b · log(1-p) + c)`. Three parameters. Generalises Platt scaling (which is the case `a = -b`). Smooth, robust on small data. Not in sklearn — use `pip install betacal`.

### Boosting

Train models sequentially; each new model corrects the residual errors of the previous ensemble. Final prediction = weighted sum. Used by XGBoost, LightGBM, HistGradientBoosting. Reduces both bias and variance. Best general-purpose tabular family.

### Bootstrap sample

Random sample with replacement of the same size as the original. Each bootstrap sample contains ~63% unique rows from the original; the remaining ~37% are duplicates of the chosen rows.

### Brier score

```
Brier = (1/N) · Σ(pᵢ - yᵢ)²
```

Mean squared error between predicted probabilities and 0/1 labels. Proper scoring rule — penalises both miscalibration and bad ranking. Lower is better. `brier_score_loss(y_true, y_prob)`.

## C

### Calibration

The property that predicted probabilities match observed frequencies. A calibrated 0.7 means: among bookings scored at 0.7, 70% actually cancel. See `CalibratedClassifierCV`, isotonic regression, Platt scaling, beta calibration, reliability diagram.

### `CalibratedClassifierCV`

Wraps a classifier in a calibration step. Internally does CV: train base model on inner folds, learn a calibration map on out-of-fold predictions. Methods: `"isotonic"` (non-parametric step function), `"sigmoid"` (Platt scaling). Critical for any application that uses the predicted probability quantitatively, not just for ranking.

### CatBoost

Tree-boosting library with native categorical handling (no encoding needed). Good when categorical features dominate. Slower training than XGBoost / LightGBM. Not currently in our pipeline; tracked as a task for evaluation.

### `class_weight="balanced"`

Down-weights the majority class during loss computation. Improves recall on minority class at the cost of probability distortion (predicted scores shift upward). We *removed* it from RF and XGBoost because it breaks calibration. Use only if you'll never use the probabilities quantitatively.

### Coefficient (linear model)

The weight `wⱼ` applied to feature `xⱼ`. For OHE'd categoricals: one coefficient per level. Interpretable as log-odds change per unit increase in `xⱼ` (after scaling). Available via `pipeline.named_steps["clf"].calibrated_classifiers_[0].estimator.coef_`.

### `ColumnTransformer`

Applies different sub-pipelines to different columns. Constructor: `ColumnTransformer([(name, transformer, columns), ...])`. Critical for tabular ML with mixed types.

### Concept drift

The relationship between features and target changes over time. Cancellation behaviour shifts as channel mix, pricing, and policy change. Random CV hides drift; temporal CV exposes it.

### `confusion_matrix(y_true, y_pred)`

2×2 table of (true_label, predicted_label) counts. `cm[0][0] = TN, cm[0][1] = FP, cm[1][0] = FN, cm[1][1] = TP`.

### `cross_validate`

```python
cross_validate(estimator, X, y, cv=cv, scoring=list, return_train_score=False, n_jobs=-1)
```

Run CV, return dict of metric arrays (`test_<metric>`, etc.). Doesn't return per-row predictions — for that use `cross_val_predict`.

### `cross_val_predict`

Returns one OOF prediction per row. Requires the CV folds to form a partition — works with `KFold`, `StratifiedKFold`. Fails with `TimeSeriesSplit` (raises `ValueError: cross_val_predict only works for partitions`). Workaround: manual loop over `cv.split()`.

### Cross-entropy / Log-loss

Loss function for probabilistic classifiers:

```
L = -Σ [y · log(p) + (1-y) · log(1-p)]
```

Penalises both wrong direction and overconfidence. The objective for logistic regression, XGBoost's `binary:logistic`, MLP.

### CV (Cross-validation)

K-fold split: train on k-1 folds, evaluate on the held-out fold, rotate, average. Quantifies model performance robustly and gives a variance estimate. See `StratifiedKFold`, `KFold`, `TimeSeriesSplit`.

## D

### Decision boundary

The hypersurface in feature space where the classifier's predicted probability crosses 0.5 (or the chosen threshold). For LogReg it's a hyperplane `w·x + b = 0`.

### Decision tree

Recursively splits the feature space by simple rules (`lead_time_days ≤ 30?`). Each leaf has a constant prediction. Overfits horribly on its own; useful only as a building block for RF / XGBoost.

### Diagnostic CV

The `StratifiedKFold` random-fold CV we run alongside `TimeSeriesSplit`. Its score is *not* the headline; the gap between it and the temporal CV tells us about drift.

### Discrete-time hazard

Conditional probability of an event happening in the next time step *given* it hasn't happened yet. Used in survival analysis. Notebook 08 implements a hazard model for cancellation, expanding bookings into (booking × snapshot) rows.

## E

### Early stopping

For iterative models (boosting, NNs), monitor a held-out validation metric during training; stop when it stops improving for `n_iter_no_change` epochs. Used by MLPClassifier (`early_stopping=True`).

### ElasticNet

Hybrid L1 + L2 regularisation. Penalty term:

```
λ · [α · Σ|wⱼ| + (1-α)/2 · Σwⱼ²]
```

where `α = l1_ratio`. The standard choice when you have many noisy or correlated features. Used by `LogisticRegression(penalty="elasticnet", l1_ratio=0.5)`. Requires `solver="saga"`.

### Embedding (entity embedding)

Learnable low-dimensional vector representation of a categorical level. State of the art for very high-cardinality categoricals in neural networks. Replaces one-hot encoding with a learned dense representation. Used in TabNet, FT-Transformer, NN-based recommendation systems.

### Ensemble

A model composed of multiple sub-models whose predictions are combined. RandomForest = ensemble of trees via averaging. XGBoost = ensemble of trees via additive boosting. Stacking = ensemble of arbitrary models via a meta-learner.

## F

### F1 score

Harmonic mean of precision and recall: `F1 = 2·P·R / (P+R)`. Single number balancing both. Range [0, 1]; perfect = 1.

### Feature engineering

Creating new features from raw data: `log_gross_amount = log1p(gross_amount)`, `is_weekend_arrival = (arrival.dayofweek >= 5)`, etc. In our project all feature engineering lives in notebook 00 so the model notebooks consume a clean, ready-to-fit parquet.

### Feature importance

Ranking of features by their contribution to the model. Multiple methods:

- **Impurity-based** (RF, XGB) — built-in. Biased toward high-cardinality features.
- **Permutation** (`permutation_importance`) — model-agnostic.
- **SHAP** — game-theoretic, decomposes into per-row contributions.

### Feature interaction

Combined effect of two features that isn't the sum of their individual effects. Trees capture interactions automatically (any split below a parent split is conditioned on the parent). Linear models don't capture interactions unless you manually add `x₁ · x₂` features.

### Frequency encoding

Replace a categorical value with its count or frequency in training. Cheap, lossy, sometimes useful for high-cardinality columns where target encoding would leak.

## G

### Gradient boosting

Boosting where each new model is fit to the *gradient* of the loss with respect to the current prediction. Equivalent to additive functional gradient descent. XGBoost, LightGBM, HistGB, CatBoost all implement variants.

## H

### `handle_unknown="ignore"`

OHE parameter: if an unseen category appears at transform time, encode as all-zero across the K columns instead of raising. Defensive default for production. Alternative: `"error"` (loud failure on unseen categories).

### Harmonic mean

`HM(a, b) = 2·a·b / (a+b)`. The basis of F1 score. Closer to the smaller of the two values than the arithmetic mean — penalises imbalance.

### HistGradientBoosting (`HistGradientBoostingClassifier`)

sklearn's native gradient boosting. Fast (histogram-based), handles NaN natively, calibrated by default. Good alternative to RF when speed matters. Tracked as a task for evaluation in our project.

### Hyperparameter

A model setting fixed before training (vs. a parameter learned from data). `C`, `l1_ratio`, `max_depth`, `n_estimators` are hyperparameters. Selected via CV or hyperparameter search.

## I

### ICE (Individual Conditional Expectation)

Per-row version of PDP. For each row, sweep the chosen feature and plot the prediction trajectory. One curve per row, overlaid. Reveals heterogeneity in feature effects.

### Imputation

Filling missing values. See `SimpleImputer`, `IterativeImputer`, `KNNImputer`. Modern alternative: use a model that handles NaN natively.

### `IterativeImputer`

Estimates missing values by iteratively modelling each column as a function of the others. Slow, occasionally helpful for highly correlated features. We use `SimpleImputer(median)` instead.

### Isotonic regression

Fits a monotone non-decreasing step function to (x, y) data. Used as a calibration method (`method="isotonic"` in `CalibratedClassifierCV`). Non-parametric, flexible, can fit any monotone curve. Needs > ~500 rows per fold for stable bins.

## K

### KFold

Random k-way split of the data. No stratification. Fine for regression, suboptimal for imbalanced classification (use `StratifiedKFold`).

### `KNNImputer`

Imputes missing values using the k-nearest non-missing rows by Euclidean distance. Slow on big data, occasionally helpful for highly structured datasets. Not used in our pipeline.

## L

### L1 regularisation (Lasso)

Penalty term `λ · Σ |wⱼ|`. Drives some coefficients to exactly zero — automatic feature selection. Unstable with correlated features.

### L2 regularisation (Ridge)

Penalty term `λ · Σ wⱼ²`. Shrinks all coefficients toward zero but rarely to exactly zero. Stable, doesn't produce sparsity.

### Lasso

Logistic / linear regression with L1 penalty. See L1 regularisation.

### Leakage

Training-time access to information that wouldn't be available at prediction time. Sources:

- Temporal: random CV on time-series data → fold sees future.
- Target: a feature derived from the target (e.g., `cancellation_fee_amount` after status is known).
- Scaling: fitting StandardScaler on the entire dataset before CV.

Pipelines prevent the third kind automatically; the first two require explicit care.

### LightGBM

Tree-boosting library similar to XGBoost. Faster training (histogram + leaf-wise growth). Worse on small data. Not in our pipeline.

### Lift@k

```
lift@k = precision_among_top_k_percent / base_rate
```

Operational metric. Lift of 3 = top-k% bookings cancel 3× more often than population average.

### `liblinear`

Fast LogReg solver supporting L1 and L2 (not ElasticNet). Doesn't support `multi_class="multinomial"`. Doesn't parallelize well. Our previous default before ElasticNet was needed.

### Logistic regression

`p = sigmoid(w · x + b)`. Models log-odds as a linear function of features. Trained by maximising log-likelihood (equivalently minimising cross-entropy). Standard baseline for binary classification.

### Log-loss / Cross-entropy loss

```
L = -Σ [y · log(p) + (1-y) · log(1-p)]
```

The objective function for logistic regression and probabilistic classifiers.

### `log_gross_amount`

Engineered feature: `log1p(gross_amount)`. Tail damping — turns the long-tailed gross_amount distribution into something approximately Gaussian, which linear models love.

## M

### `max_depth`

Tree hyperparameter. Limits how deep individual trees can grow. RF: high values (None, 20+) → individual trees overfit but average smooths it; lower values (8-14) → cleaner, faster scoring. XGBoost: typically 4-8 (deep trees boost less efficiently). MLP: doesn't apply.

### `max_features` (RF)

Number of features considered at each tree split. `"sqrt"` for classification (default); `"log2"` is similar. Smaller = more decorrelation between trees.

### `MLPClassifier`

sklearn's multi-layer perceptron. Feed-forward NN with `hidden_layer_sizes` neurons per layer, ReLU activation. Not state-of-the-art on tabular data — included as a sanity floor against the tree models.

### `min_samples_leaf`

Tree hyperparameter. Each leaf must contain at least this many training rows. Prevents memorisation of individual rows. RF: typically 10-50. XGBoost equivalent: `min_child_weight`.

## N

### `n_estimators`

Number of trees in RF or boosting rounds in XGBoost. RF: 200-500 is plenty (diminishing returns past that). XGBoost: 300-1000 typically, tuned together with `learning_rate` (smaller LR needs more trees).

### `n_jobs`

Parallelism. `-1` = use all CPU cores. Watch out for nested parallelism — if you set both an outer `cross_validate(n_jobs=-1)` and an inner model's `n_jobs=-1`, you can deadlock or thrash. We mostly let CV parallelise and keep model `n_jobs` modest.

### Naive Bayes

Probabilistic classifier assuming feature independence given the class. Famously badly calibrated. Not in our pipeline.

## O

### OOF (Out-of-Fold) predictions

Predictions on rows that were *not* in the training fold of the model that produced them. Used for honest threshold tuning, stacking, and calibration. Obtained via `cross_val_predict` for partitioning CVs, or manually for `TimeSeriesSplit`.

### OneHotEncoder

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

K levels → K binary columns. The standard categorical encoding for ≤ ~30 levels.

### OrdinalEncoder

Maps categorical levels to integers 0, 1, 2, ... Useful when there's a real order (`stay_bucket = short < mid < long`). False ordering imposed on nominal features hurts linear models, less so trees.

## P

### Partial Dependence Plot (PDP)

Average prediction as one feature is swept over its range, marginalising over the rest of the data. Reveals the model's marginal response to a feature. Assumes feature independence — biased when features are correlated.

### `PartialDependenceDisplay.from_estimator`

```python
PartialDependenceDisplay.from_estimator(pipeline, X, features=[...], kind="both", ...)
```

`kind="average"` = PDP only. `kind="individual"` = ICE only. `kind="both"` = both overlaid.

### `permutation_importance`

```python
permutation_importance(pipeline, X_test, y_test, n_repeats=5, scoring="roc_auc")
```

Shuffle one feature's column, observe score drop. Returns mean and per-repeat distribution. Model-agnostic. Computationally expensive.

### Pipeline

Sequence of (name, transformer) steps applied in order. Last step is the classifier; previous are preprocessors. The container that prevents leakage during CV by re-fitting each step on each training fold.

### Platt scaling

A calibration method: `p_calibrated = sigmoid(a · raw + b)`. Two parameters. Simple, robust on small data. Used as `method="sigmoid"` in `CalibratedClassifierCV`. Fails when the miscalibration isn't sigmoid-shaped.

### `predict` vs `predict_proba`

`predict` returns hard labels (0 or 1) using a threshold of 0.5. `predict_proba` returns `[P(y=0), P(y=1)]` arrays. Always use `predict_proba` for evaluation and threshold tuning.

### Precision

```
P = TP / (TP + FP)
```

"When the model says positive, how often is it right?" Lower bound: 0; upper bound: 1. Imbalanced classification's headline alongside recall.

### `precision_recall_curve(y_true, y_prob)`

Returns `(precision_array, recall_array, thresholds_array)` sweeping the threshold from 1 (everything predicted negative) to 0 (everything predicted positive). Note: precision and recall arrays are length N, thresholds is length N-1 (the highest threshold has no `prec/rec` value because no positives are predicted).

### Probability calibration

See "Calibration".

## R

### Random Forest

Ensemble of decision trees on bootstrap samples + feature subsampling. Average their predicted probabilities. Implemented as `RandomForestClassifier`.

### Random state

`random_state=42` everywhere. Reproducibility — same data + same random_state = same result. Doesn't affect generalisation.

### `RareCategoryCollapser` (custom)

Our custom transformer. Categories with count below `min_count` get folded into `"Other"`. Used for `primaryGuest_address_countryCode` (182 levels). See §8.1.

### Recall (sensitivity, TPR)

```
R = TP / (TP + FN)
```

"When the truth is positive, how often does the model catch it?" Tracked against precision in PR curves.

### Regularisation

Adding a penalty to the loss to keep coefficients small. L1, L2, ElasticNet. Prevents overfitting on high-dimensional data.

### ReLU

`f(x) = max(0, x)`. The default activation for hidden layers in MLP / deep nets. Cheap, doesn't saturate (unlike sigmoid), works.

### Reliability diagram (calibration plot)

Bin predictions by score; within each bin compute mean predicted probability vs mean true label. Plot the latter against the former. Perfectly calibrated model = points on the diagonal.

### `remainder='drop'`

`ColumnTransformer` parameter. Columns not listed in any transformer are dropped. Alternative `'passthrough'` keeps them.

### Ridge

Logistic / linear regression with L2 penalty. See L2 regularisation.

### `roc_auc_score(y_true, y_prob)`

Computes AUC. Use `y_prob` (probabilities), not `y_pred` (labels).

### `roc_curve(y_true, y_prob)`

Returns `(fpr_array, tpr_array, thresholds_array)` for plotting the ROC curve.

## S

### `saga` solver

LogReg solver supporting all penalties (L1, L2, ElasticNet, None). Slower than `liblinear` but more general. Required for ElasticNet.

### `scale_pos_weight`

XGBoost parameter; equivalent of `class_weight`. Removed from our pipeline because it distorts probabilities.

### SHAP (SHapley Additive exPlanations)

Game-theoretic per-feature contribution decomposition for individual predictions. Sum of contributions = prediction − base_rate. Plots: waterfall, force, beeswarm, bar, scatter. Different explainers per model family (`TreeExplainer`, `KernelExplainer`, `LinearExplainer`, `DeepExplainer`).

### Sigmoid (logistic function)

```
sigmoid(z) = 1 / (1 + e⁻ᶻ)
```

S-shaped, maps ℝ → (0, 1). The link function for logistic regression. Saturates at extremes — derivative is tiny when z is far from zero.

### `SimpleImputer`

```python
SimpleImputer(strategy="median")
SimpleImputer(strategy="most_frequent")
SimpleImputer(strategy="constant", fill_value="MISSING")
```

Replaces NaN with a column statistic computed during `fit`.

### Sparse matrix

Compact representation for matrices with mostly-zero entries. `scipy.sparse`. OHE with thousands of columns benefits; for our scale we use dense (`sparse_output=False`).

### `sparse_threshold` (ColumnTransformer)

If the combined output of all transformers has < 30% non-zero, return sparse; else dense. We're dense.

### Stacking

Ensemble method: train multiple base models, then train a meta-model on their out-of-fold predictions. Often beats simple averaging when base models have different error patterns. Tracked as a task to compare against the mean ensemble in notebook 05.

### `StandardScaler`

```python
z = (x - μ_train) / σ_train
```

Subtract mean, divide by std. Computed from training data only.

### `stratify` (in train_test_split)

Preserve class proportions across splits. Without it, a 21%-positive dataset could have a 14%-positive test fold by chance. Always use `stratify=y` for imbalanced classification.

### `StratifiedKFold`

K-fold CV preserving class proportions in each fold. The standard CV for imbalanced classification. `shuffle=True` randomises row order before splitting.

## T

### Target encoding

Replace a categorical level with the smoothed mean of the target within that level (computed on training, OOF for safety). Big signal density gain over OHE for high-cardinality columns. `sklearn.preprocessing.TargetEncoder` (v1.3+) handles OOF internally.

### Temporal split

Train on rows older than some cutoff; test on newer. The honest evaluation framework for time-series data. We use `is_temporal_test` from notebook 00 §7.

### `TimeSeriesSplit(n_splits)`

Walk-forward CV. Fold k trains on `[0..k/n]` and validates on `[k/n..(k+1)/n]`. Mimics deployment. Not a partition — breaks `cross_val_predict`.

### Threshold

The probability cutoff above which a prediction becomes "positive". Default 0.5; usually suboptimal for imbalanced data. Tune via OOF training predictions, then apply once to the test set.

### `TopKCategoryCollapser` (historical custom)

Older version of `RareCategoryCollapser`. Kept top-k by frequency. We replaced it with `RareCategoryCollapser(min_count=10)` because count-thresholding is more principled.

### Tree (decision tree)

Recursive partition of the feature space. Each internal node: a feature + a threshold. Each leaf: a prediction. Trained greedily by impurity reduction.

## U

### Univariate transform

A transform applied independently to each column. `StandardScaler`, `SimpleImputer`, `OneHotEncoder` are all univariate (each column processed separately). `IterativeImputer` is multivariate (uses other columns to impute one).

## V

### Validation set

Held-out data used during training to monitor performance (early stopping) or after training to tune hyperparameters / thresholds. Distinct from the test set, which is used exactly once for final reporting.

### `verbose_feature_names_out` (ColumnTransformer)

`True` (default): prefix output columns with `<transformer_name>__`. `False`: keep unprefixed names. We use `False` for cleaner SHAP plot labels.

## X

### XGBoost (`XGBClassifier`)

Gradient-boosted decision trees. Industry standard for tabular ML. Key parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, `tree_method`.

### XAI (Explainable AI)

Methods for understanding why a model made a prediction. We use permutation importance (model-agnostic), PDP + ICE (response curves), and SHAP (per-row decomposition).

## Y

### `y_pred` vs `y_prob`

`y_pred` = hard labels (0/1) after thresholding. `y_prob` = probabilities. Use `y_prob` for AUC, AP, Brier, calibration; use `y_pred` for confusion matrix, precision, recall, F1.

---

# Quick reference: when to reach for what

| Question | What to look at |
|---|---|
| "Is my model better than random?" | AUC (test set, temporal hold-out) |
| "How does it do on the positives specifically?" | Average precision, lift@10% |
| "Are my probabilities trustworthy?" | Reliability diagram + Brier score |
| "Where should I set the decision threshold?" | Threshold sweep table; OOF-tuned F1-optimal or precision-first |
| "Why did the model say this booking is risky?" | SHAP waterfall on that row |
| "Which features matter overall?" | SHAP beeswarm, permutation importance |
| "How does the model react to lead_time?" | PDP + ICE for `lead_time_days`; SHAP dependence scatter |
| "Does the model work for property X?" | Per-property AUC in §8 |
| "Is the temporal split lying?" | Compare temporal-test AUC vs random-test diagnostic AUC |
| "Did calibration help?" | Reliability diagram pre/post; Brier score before/after |
| "Why is CV slow?" | Calibration multiplier — use stripped `cv_pipeline` |
| "Why did `cross_val_predict` fail?" | TimeSeriesSplit isn't a partition — use the manual OOF loop |

---

# Decisions you've locked in vs. open

**Locked (won't touch unless evidence forces it).**

- Target = binary `is_cancelled` (1 = Canceled, 0 = CheckedOut/InHouse).
- Temporal split via `is_temporal_test`.
- LogReg with ElasticNet, `l1_ratio=0.5`, `solver="saga"`.
- ColumnTransformer routing numeric / low-card / high-card.
- `RareCategoryCollapser(min_count=10)` for countryCode.
- Calibration with `CalibratedClassifierCV(method="isotonic", cv=5)`.
- Dual CV: `TimeSeriesSplit(3)` + `StratifiedKFold(3)`.
- Threshold tuning on OOF predictions.
- Three operating points (F1, precision-first, recall-first).
- Per-property + per-channel + per-rateplan + per-lead-time slice analysis.
- XAI: permutation importance (bar + box), PDP + ICE, SHAP (waterfall + beeswarm + bar + scatter).

**Open (tracked, awaiting evidence to flip).**

- Switch `stay_bucket` from OneHotEncoder to `OrdinalEncoder` (true ordering).
- Switch `countryCode` from `RareCategoryCollapser` to `TargetEncoder`.
- Try HistGradientBoosting / CatBoost vs current RF / XGBoost.
- Add prediction intervals to per-property heatmap (conformal prediction).
- Investigate `ratePlan_type` (6 buckets) vs `ratePlan_category` (59 buckets).
- Hyperparameter search via `RandomizedSearchCV` for each model.
- Move daily scoring from static rescoring to discrete-time hazard if notebook 08 wins by > 0.02 AUC at d ≤ 7.
