# ---------------------------------------------------------------------------
# src/hazard.py
# Discrete-time cancellation HAZARD model (notebook 08), made serveable +
# retrainable from src so the app and the retrain API can use it like the
# static models.
#
# What it does
#   * expands bookings into a person-period grid (one row per day-before-arrival
#     snapshot) and learns h_d = P(cancel in window ending at d | survived, x);
#   * calibrates with PER-SNAPSHOT-BAND isotonic maps (daily d<=14 vs the coarse
#     tail) — a wide-window hazard is not on the same probability scale as a daily
#     one, so one pooled map over heterogeneous widths miscalibrates both;
#   * serves a per-booking P(cancel before arrival) via the survival product
#     P_cum(D) = 1 - Π_{u=1..D} (1 - h_u), evaluated on a FRESH forward grid
#     u=1..D (D = the booking's current days_until_arrival) — the v11 bugfix;
#   * pins scoring categoricals to the EXACT train dtype (unseen -> NaN), the
#     other v11 fix (XGBoost 2.x recodes the declared category array by name).
#
# Heavy deps (xgboost / sklearn) are imported lazily; the person-period build and
# the survival product are pure pandas/numpy and unit-testable on their own.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import Final

import numpy as np
import pandas as pd

from . import walkforward as wf
from .features import load_feature_roster
from .paths import data_dir, repo_root

SEED: Final[int] = 42
AXIS: Final[str] = "days_until_arrival"
ARRIVAL: Final[str] = "arrival"           # booking arrival-date column
SNAP_FINE: Final[list[int]] = list(range(1, 15))                  # daily, decision-relevant horizon
# Coarse train-only tail — extended to ~270d so LONG-LEAD cancellations are not
# mislabelled as survivors. At max_snap=90 we missed ~4.5% of cancels (event_d>90:
# (90,120]=868, (120,180]=515, (180,270]=219). Data supports it (lead>=270 ~0.2%
# of bookings; >270 cancels ~0.1% — negligible residual).
SNAP_COARSE: Final[list[int]] = [21, 30, 45, 60, 90, 120, 180, 270]
SNAP: Final[list[int]] = sorted(SNAP_FINE + SNAP_COARSE)
# Calibration band edge: daily snapshots (<=14, width 1) vs the coarse tail
# (>14, widths 7..90). Isotonic is fit separately per band (see fit_hazard) —
# pooling a wide-window hazard with a daily one miscalibrates both.
CAL_BAND_EDGE: Final[int] = max(SNAP_FINE)   # = 14

HAZARD_PATH: Final[str] = "08_hazard_model.joblib"
HAZARD_CARD: Final[str] = "reports/tables/08_hazard/model_card.json"

# HP grid mirrors notebook 08 (searched on the temporal val by AP).
HP_GRID: Final[list[dict]] = [
    dict(max_depth=6, learning_rate=0.05, min_child_weight=5,  reg_lambda=1.0,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, learning_rate=0.05, min_child_weight=20, reg_lambda=5.0,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, learning_rate=0.05, min_child_weight=10, reg_lambda=5.0,  subsample=0.9, colsample_bytree=0.7),
    dict(max_depth=7, learning_rate=0.04, min_child_weight=30, reg_lambda=8.0,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.08, min_child_weight=20, reg_lambda=5.0,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, learning_rate=0.03, min_child_weight=50, reg_lambda=10.0, subsample=0.7, colsample_bytree=0.9),
]


# =============================================================================
# Feature lists + event columns
# =============================================================================
def feature_lists(clean: pd.DataFrame) -> tuple[list[str], list[str]]:
    r = load_feature_roster()
    num = [c for c in r["numeric"] if c in clean.columns]
    cat = [c for c in r["categorical"] if c in clean.columns]
    return num, cat


def add_event_columns(clean: pd.DataFrame) -> pd.DataFrame:
    """Add `lead`, `event_d` (cancel timing for events else NaN) and `src_idx`.

    Matches 00's target: event = status==1 AND cancel_days_before_arrival>=0
    (pre-arrival or same-day cancels); event_d is the days-before-arrival timing;
    non-events (incl. no-shows) are censored survivors. Keeps lead>0.
    """
    out = clean.copy()
    arr = pd.to_datetime(out["arrival"], utc=True, errors="coerce")
    cre = pd.to_datetime(out["created"], utc=True, errors="coerce")
    out["lead"] = (arr - cre) / pd.Timedelta(days=1)
    status = wf.target_series(out)
    cdba = pd.to_numeric(out.get("cancel_days_before_arrival"), errors="coerce")
    # Event = pre-arrival OR same-day cancel (cdba >= 0), matching 00's target.
    # Clip to a tiny positive so an exactly-on-arrival cancel (cdba == 0) lands in
    # the first daily window (0, 1] instead of being lost at the boundary.
    is_event = status.eq(1) & cdba.ge(0)
    out["event_d"] = np.where(is_event, np.clip(cdba, 1e-6, None), np.nan)
    out = out[out["lead"] > 0].reset_index(drop=True)
    out["src_idx"] = out.index
    return out


# =============================================================================
# Person-period expansion (pure pandas — unit-testable)
# =============================================================================
def build_person_period(clean: pd.DataFrame, num: list[str], cat: list[str],
                         *, cat_dtypes: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Expand bookings into the person-period hazard grid (nb08 logic).

    `clean` must already have `lead`, `event_d`, `src_idx` (see add_event_columns).
    Categoricals are pinned to `cat_dtypes` if given (train dtype; unseen -> NaN),
    otherwise inferred here and the resulting dtypes are returned — so the caller
    can reuse the TRAIN dtypes when building val/test grids (the v11 fix).
    """
    prevs = {d: (SNAP[i - 1] if i > 0 else 0) for i, d in enumerate(SNAP)}
    lead = clean["lead"].to_numpy(); ev = clean["event_d"].to_numpy(); idx = clean["src_idx"].to_numpy()
    parts = []
    for d in SNAP:
        dn = prevs[d]
        at_risk = (lead >= d) & (np.isnan(ev) | (ev <= d))         # exists & not already cancelled
        y = ((~np.isnan(ev)) & (ev > dn) & (ev <= d)).astype(int)  # cancelled within (dn, d]
        parts.append(pd.DataFrame({"src_idx": idx[at_risk], AXIS: float(d),
                                   "width": d - dn, "y": y[at_risk]}))
    pp = pd.concat(parts, ignore_index=True)
    pp = pp.merge(clean[["src_idx"] + num + cat], on="src_idx", how="left")

    if cat_dtypes is None:
        for c in cat:
            pp[c] = pp[c].astype("category")
        cat_dtypes = {c: pp[c].dtype for c in cat}
    else:
        for c in cat:
            pp[c] = pp[c].astype(cat_dtypes[c])                    # pin to train dtype; unseen -> NaN
    for c in num + [AXIS]:
        pp[c] = pp[c].astype("float64")
    return pp, cat_dtypes


# =============================================================================
# Survival product (pure — testable with a stub hazard fn)
# =============================================================================
def survival_cancel_proba(bookings: pd.DataFrame, haz_fn, num: list[str], cat: list[str],
                          cat_dtypes: dict, *, horizon_col: str = AXIS,
                          snaps: list[int] | None = None) -> np.ndarray:
    """Per-booking P(cancel before arrival) = 1 - Π_{s∈snaps, s≤D} (1 - h_s).

    The grid is the model's TRAINED snapshot grid (daily 1..14 + the coarse tail),
    NOT a daily grid capped at 14 — so LONG-LEAD bookings (arrival far out) accumulate
    cancel risk across the full horizon the model knows (the #5 fix). D = the
    booking's `horizon_col` (days_until_arrival), capped at its `lead`. Pass
    `snaps=hz["snap"]` so scoring matches exactly what the model was trained on.
    `haz_fn(X)->h` lets tests pass a stub. Short-lead is unchanged (snaps≤14 = daily).
    """
    snaps_arr = np.array(sorted(snaps if snaps is not None else SNAP), dtype=int)
    b = bookings.reset_index(drop=True).copy()
    n = len(b)
    D = np.floor(pd.to_numeric(b[horizon_col], errors="coerce").fillna(0)).astype(int).to_numpy()
    if "lead" in b.columns:
        D = np.minimum(D, np.floor(pd.to_numeric(b["lead"], errors="coerce").fillna(0)).astype(int).to_numpy())
    if n == 0 or snaps_arr.size == 0:
        return np.zeros(n, dtype=float)
    # Cross bookings × snapshots, keep the snapshots each booking actually traverses
    # (snap ≤ its days-to-arrival). Vectorised — no per-booking Python loop.
    row_idx = np.repeat(np.arange(n), snaps_arr.size)
    snap_col = np.tile(snaps_arr, n)
    keep = snap_col <= D[row_idx]
    if not keep.any():
        return np.zeros(n, dtype=float)
    g = b.iloc[row_idx[keep]].reset_index(drop=True).copy()
    g["_row"] = row_idx[keep]
    g[AXIS] = snap_col[keep].astype("float64")
    for c in cat:
        g[c] = g[c].astype(cat_dtypes[c])                          # pin to train dtype
    for c in num + [AXIS]:
        g[c] = g[c].astype("float64")
    FEATS = num + [AXIS] + cat
    g["_h"] = np.clip(haz_fn(g[FEATS]), 0, 0.999999)
    g = g.sort_values(["_row", AXIS])
    g["_pcum"] = 1 - g.groupby("_row")["_h"].transform(lambda s: (1 - s).cumprod())
    last = g.groupby("_row")["_pcum"].last()                       # cumulative at the largest snap ≤ D
    out = np.zeros(n, dtype=float)
    out[last.index.to_numpy()] = last.to_numpy()
    return out


# =============================================================================
# Fit + calibrate (xgboost — kernel-validated)
# =============================================================================
def _sample_hp(rng) -> dict:
    """Sample one hyperparameter config (depth, lr, min_child_weight, reg_lambda,
    subsample, colsample) — the RandomizedSearch space for the hazard model."""
    return dict(
        max_depth=int(rng.choice([4, 5, 6, 7, 8])),
        learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.15)))),
        min_child_weight=int(rng.choice([5, 10, 20, 30, 50])),
        reg_lambda=float(np.exp(rng.uniform(np.log(1.0), np.log(15.0)))),
        subsample=float(rng.uniform(0.7, 0.95)),
        colsample_bytree=float(rng.uniform(0.7, 0.95)),
    )


def fit_hazard(clean_resolved: pd.DataFrame, *, val_frac: float = 0.15,
               n_iter: int = 15, seed: int = SEED, fixed_hp: dict | None = None) -> dict:
    """Fit the hazard model on RESOLVED bookings via a RandomizedSearch (with
    EARLY STOPPING — no fixed tree count) + PER-SNAPSHOT-BAND isotonic calibration,
    both on a temporally held-out (most-recent-by-created) validation slice.

    `fixed_hp` skips the search and fits ONE model with the given hyperparameters (still
    early-stopped + calibrated). This is the FROZEN-hp path used by leak-free per-fold
    evaluation / refit (src.model_eval), so a walk-forward doesn't re-tune per fold — the
    same discipline the static models use via training._card_hp. None => full search.

    Returns a dict artifact: {model, iso, iso_bands, num, cat, cat_dtypes, snap,
    axis, hp, val_ap, best_iteration, n_train_pp}. `iso` is the pooled map (kept
    as a fallback); `iso_bands` = {edge, le, gt} holds the daily/coarse maps.
    """
    from xgboost import XGBClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score

    ce = add_event_columns(clean_resolved)
    num, cat = feature_lists(ce)
    created = pd.to_datetime(ce["created"], utc=True, errors="coerce")
    cutoff = created.quantile(1 - val_frac)
    tr_books = ce[created <= cutoff]; va_books = ce[created > cutoff]

    pp_tr, cat_dtypes = build_person_period(tr_books, num, cat)
    pp_va, _ = build_person_period(va_books, num, cat, cat_dtypes=cat_dtypes)
    FEATS = num + [AXIS] + cat
    Xtr, ytr = pp_tr[FEATS], pp_tr["y"].to_numpy()
    Xva, yva = pp_va[FEATS], pp_va["y"].to_numpy()

    rng = np.random.RandomState(seed)
    if fixed_hp is not None:
        candidates = [fixed_hp]                                    # frozen-hp fast path (1 fit)
    else:
        # baseline config first, then n_iter random samples (RandomizedSearch).
        candidates = [HP_GRID[0]] + [_sample_hp(rng) for _ in range(n_iter)]
    best = None
    for hp in candidates:
        m = XGBClassifier(n_estimators=2000, tree_method="hist", enable_categorical=True,
                          eval_metric="aucpr", early_stopping_rounds=50,
                          objective="binary:logistic", n_jobs=-1, random_state=seed, **hp)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, hp, m)
    val_ap, hp, model = best
    # Per-snapshot-band isotonic calibration. A wide-window (coarse-tail) hazard
    # and a daily hazard live on different probability scales, so ONE pooled map
    # over mixed widths miscalibrates both. Fit a map for the daily band
    # (d<=CAL_BAND_EDGE) and one for the coarse tail; fall back to the pooled map
    # for any band too sparse to fit a stable isotonic on.
    raw_va  = model.predict_proba(Xva)[:, 1]
    axis_va = pp_va[AXIS].to_numpy()
    iso_pooled = IsotonicRegression(out_of_bounds="clip").fit(raw_va, yva)
    def _fit_band(mask):
        if int(mask.sum()) >= 500 and np.unique(yva[mask]).size > 1:
            return IsotonicRegression(out_of_bounds="clip").fit(raw_va[mask], yva[mask])
        return iso_pooled
    iso_bands = {"edge": CAL_BAND_EDGE,
                 "le": _fit_band(axis_va <= CAL_BAND_EDGE),
                 "gt": _fit_band(axis_va >  CAL_BAND_EDGE)}
    return {"model": model, "iso": iso_pooled, "iso_bands": iso_bands,
            "num": num, "cat": cat, "cat_dtypes": cat_dtypes,
            "snap": SNAP, "axis": AXIS, "hp": hp, "val_ap": float(val_ap),
            "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
            "n_train_pp": int(len(pp_tr))}


def hazard_fn(hz: dict):
    """Return a calibrated hazard callable haz(X)->h from a fitted artifact.

    Applies PER-SNAPSHOT-BAND isotonic calibration: rows with days_until_arrival
    (AXIS) <= edge use the daily-band map, the rest the coarse-tail map. AXIS is
    always part of the scoring FEATS (survival_cancel_proba guarantees it), so the
    band is known per row. Falls back to the single pooled map for older artifacts
    that predate `iso_bands`.
    """
    model = hz["model"]
    bands = hz.get("iso_bands")
    if not bands:                                  # backward-compat: pooled map only
        iso = hz["iso"]
        return lambda X: iso.predict(model.predict_proba(X)[:, 1])
    edge, iso_le, iso_gt = bands["edge"], bands["le"], bands["gt"]

    def haz(X):
        raw = model.predict_proba(X)[:, 1]
        d = np.asarray(X[AXIS], dtype=float)
        return np.where(d <= edge, iso_le.predict(raw), iso_gt.predict(raw))
    return haz


def score_upcoming_hazard(hz: dict, bookings: pd.DataFrame) -> np.ndarray:
    """Per-booking P(cancel before arrival) for upcoming bookings (survival product
    over the model's full trained snapshot grid — long-lead bookings included)."""
    return survival_cancel_proba(bookings, hazard_fn(hz), hz["num"], hz["cat"],
                                 hz["cat_dtypes"], snaps=hz.get("snap"))


# =============================================================================
# Persist / load
# =============================================================================
def save_hazard(hz: dict, path=None) -> str:
    import joblib
    p = data_dir() / HAZARD_PATH if path is None else path
    joblib.dump(hz, p)
    return str(p)


def load_hazard(path=None) -> dict:
    import joblib
    p = data_dir() / HAZARD_PATH if path is None else path
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — train the hazard model first (retrain_hazard).")
    return joblib.load(p)


def hazard_available() -> bool:
    return (data_dir() / HAZARD_PATH).exists()


# =============================================================================
# Retrain (point-in-time) — fit on all resolved data, persist
# =============================================================================
def retrain_hazard(*, asof=None, persist: bool = True, seed: int = SEED,
                   refresh_eval: bool = False) -> dict:
    """Fit the deployment hazard model on ALL data resolved by `asof` and persist.

    `refresh_eval=True` rebuilds the Model-Performance page's eval artifact afterwards
    (Data/model_eval_hazard.parquet) so the page tracks the freshly deployed model."""
    from . import load_clean_reservations
    clean = wf.add_outcome_known_date(load_clean_reservations())
    known = pd.to_datetime(clean[wf.KNOWN_COL], utc=True, errors="coerce")
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof is not None else pd.Timestamp(known.max())
    resolved = clean[known <= asof_ts]

    hz = fit_hazard(resolved, seed=seed)
    result = {"model": "hazard", "asof": str(asof_ts.date()), "val_ap": hz["val_ap"],
              "hp": hz["hp"], "n_train_pp": hz["n_train_pp"], "n_books_resolved": int(len(resolved))}
    if persist:
        jp = save_hazard(hz)
        card = {"model": "hazard", "retrained_at": pd.Timestamp.utcnow().isoformat(),
                "asof": str(asof_ts.date()), "val_ap": hz["val_ap"], "hyperparams": hz["hp"],
                "n_train_person_period": hz["n_train_pp"], "snap": SNAP, "axis": AXIS,
                "features_numeric": hz["num"], "features_categorical": hz["cat"]}
        cp = repo_root() / HAZARD_CARD
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(card, indent=2))
        result["persisted"] = {"joblib": jp, "card": str(cp)}

    if refresh_eval:
        try:
            from . import model_eval as _me
            _me.model_eval("hazard", refresh=True)
            result["eval_refreshed"] = True
        except Exception as e:  # noqa: BLE001
            result["eval_refresh_error"] = str(e)[:120]
    return result


# =============================================================================
# Per-arrival-night aggregation + calibration (statistically correct)
# =============================================================================
def per_night_table(bookings: pd.DataFrame, cancel_proba, *, arrival_col: str = ARRIVAL,
                    hotel_col: str | None = None, label=None) -> pd.DataFrame:
    """Aggregate per-booking cancel probabilities to per-(arrival-night) EXPECTED
    freed rooms, with the Poisson-binomial variance Sum p(1-p). If `label` (0/1
    actual cancel-before-arrival) is given, also returns the actual freed count.

    Columns: arrival_date [, hotel], n, exp (=Sum p), var (=Sum p(1-p)) [, act].
    """
    df = pd.DataFrame({"arrival_date": pd.to_datetime(bookings[arrival_col], utc=True).dt.date,
                       "p": np.asarray(cancel_proba, dtype=float)})
    if hotel_col is not None and hotel_col in bookings:
        df["hotel"] = bookings[hotel_col].to_numpy()
    if label is not None:
        df["act"] = np.asarray(label, dtype=int)
    df["var"] = df["p"] * (1.0 - df["p"])
    keys = ["arrival_date"] + (["hotel"] if "hotel" in df.columns else [])
    agg = {"n": ("p", "size"), "exp": ("p", "sum"), "var": ("var", "sum")}
    if "act" in df.columns:
        agg["act"] = ("act", "sum")
    return df.groupby(keys).agg(**agg).reset_index()


def recalibration_factor(per_night_val: pd.DataFrame) -> float:
    """Aggregate-bias fix: single multiplicative r = Sum(actual)/Sum(expected) fit
    on VALIDATION nights; apply to test expected-freed."""
    e = float(per_night_val["exp"].sum())
    return float(per_night_val["act"].sum() / e) if e > 0 else 1.0


def coverage_report(per_night: pd.DataFrame, *, levels=(0.5, 0.8, 0.9, 0.95),
                    min_n: int = 5) -> dict:
    """Per-night interval coverage under the independence (Poisson-binomial) model
    plus the overdispersion factor phi = mean[(act-exp)^2 / Sum p(1-p)]. phi>1 =>
    positively-correlated cancellations; inflate sd by sqrt(phi). Returns coverage
    at each nominal level (raw + sqrt(phi)-adjusted) and the aggregate bias %."""
    # two-sided z for common nominal levels (avoids a scipy dependency).
    _Z = {0.5: 0.6745, 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    nb = per_night[per_night["n"] >= min_n].copy()
    if not len(nb) or "act" not in nb.columns:
        return {}
    resid2 = (nb["act"] - nb["exp"]) ** 2
    phi = float((resid2 / nb["var"].replace(0, np.nan)).mean())
    out = {"phi": phi, "nights": int(len(nb)),
           "bias_pct": float(100 * (nb["exp"].sum() - nb["act"].sum()) / max(nb["act"].sum(), 1))}
    sd = np.sqrt(nb["var"]); sd_adj = np.sqrt(nb["var"] * max(phi, 1e-9))
    for lvl in levels:
        z = _Z.get(round(float(lvl), 2), 1.9600)
        out[f"cov{int(lvl*100)}"] = float(((nb["act"] >= nb["exp"] - z*sd) & (nb["act"] <= nb["exp"] + z*sd)).mean())
        out[f"cov{int(lvl*100)}_adj"] = float(((nb["act"] >= nb["exp"] - z*sd_adj) & (nb["act"] <= nb["exp"] + z*sd_adj)).mean())
    return out


# =============================================================================
# Matched, decision-time walk-forward: hazard vs static on the SAME estimand
# =============================================================================
def walk_forward_eval_hazard(*, n_folds: int = 6, horizon_days: int = 14, step_days: int = 14,
                             compare_static: bool = True, seed: int = SEED) -> dict:
    """Decision-time walk-forward for the hazard model, matched to the decision.

    Each test booking is graded once, at its decision-time horizon d = min(lead,
    `horizon_days`) (src.walkforward.make_folds). Per fold: fit the hazard model on
    bookings resolved before the window, score the test bookings via the SURVIVAL
    PRODUCT over d, and grade on cancel-by-arrival. If `compare_static`, the best
    static model is scored on the SAME rows + SAME label so the comparison targets
    the same estimand (P(cancel by arrival | open at the decision date)). The
    promotion signal uses mean(delta_AUC) > its own std (signal beats noise), not a
    magic 0.01 gate.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from . import scoring as sc, walkforward as wf, load_clean_reservations

    clean = wf.add_outcome_known_date(load_clean_reservations())
    folds = wf.make_folds(clean, n_folds=n_folds, horizon_days=horizon_days, step_days=step_days)
    static_pipe = None
    if compare_static:
        try:
            static_pipe = sc.load_model(sc.best_model())
            snum, scat = sc.model_feature_lists()
        except Exception:  # noqa: BLE001
            static_pipe = None

    rows = []
    for f in folds:
        tr, te = clean.iloc[f.train_idx], clean.iloc[f.test_idx]
        if len(tr) < 500 or len(te) < 50:
            continue
        hz = fit_hazard(tr, seed=seed)
        S = pd.Timestamp(f.origin)
        teb = te.copy()
        arr = pd.to_datetime(teb[ARRIVAL], utc=True); cre = pd.to_datetime(teb["created"], utc=True)
        teb["lead"] = (arr - cre) / pd.Timedelta(days=1)
        # Decision-time horizon: each booking is scored at d = min(lead, H) days out
        # (long-lead -> H; short-lead -> its lead), matching src.walkforward.make_folds.
        teb[AXIS] = np.minimum(teb["lead"], horizon_days).clip(lower=1)
        p_haz = survival_cancel_proba(teb, hazard_fn(hz), hz["num"], hz["cat"],
                                      hz["cat_dtypes"], snaps=hz.get("snap"))
        y = (wf.target_series(teb).to_numpy() == 1)
        row = {"fold": f.k, "S": str(S.date()), "n_test": int(len(te)),
               "auc_haz": roc_auc_score(y, p_haz) if len(set(y)) > 1 else float("nan"),
               "ap_haz": average_precision_score(y, p_haz),
               "brier_haz": brier_score_loss(y, p_haz)}
        if static_pipe is not None:
            try:
                p_st = static_pipe.predict_proba(teb[snum + scat])[:, 1]
                row["auc_static"] = roc_auc_score(y, p_st) if len(set(y)) > 1 else float("nan")
                row["ap_static"] = average_precision_score(y, p_st)
                row["delta_auc"] = row["auc_haz"] - row["auc_static"]
            except Exception as e:  # noqa: BLE001
                row["static_err"] = str(e)[:60]
        rows.append(row)

    pf = pd.DataFrame(rows)
    agg = {m: {"mean": float(pf[m].mean()), "std": float(pf[m].std())}
           for m in ["auc_haz", "ap_haz", "brier_haz", "auc_static", "ap_static", "delta_auc"]
           if m in pf.columns}
    gate = None
    if "delta_auc" in pf.columns and len(pf) > 1:
        d = pf["delta_auc"]
        gate = {"mean_delta_auc": float(d.mean()), "std": float(d.std()),
                "hazard_better": bool(d.mean() > d.std() and d.mean() > 0)}
    return {"per_fold": rows, "aggregate": agg, "gate": gate}


# =============================================================================
# Time-resolved (per-snapshot) evaluation — does the hazard track risk over time?
# =============================================================================
# The walk-forward above grades each booking ONCE (its decision horizon). To test
# whether the hazard captures how risk CHANGES as arrival approaches, we score each
# TEST booking at EVERY snapshot it traverses and measure discrimination +
# calibration PER days-until-arrival. Leak-free: `hz` is trained on data resolved
# before the test bookings.
def per_snapshot_scores(hz: dict, clean_test: pd.DataFrame,
                        *, max_days: int | None = None) -> pd.DataFrame:
    """Long frame [days_until_arrival, width, y, p] — the CALIBRATED hazard for each
    (test booking × snapshot). Feed a per-fold `hz` and that fold's TEST bookings.

    `max_days` caps the evaluated horizon. A decision-time test set is open at
    S* = arrival - H, so it is survivor-selected and CANNOT contain events beyond
    the daily band (all y=0 for d>H) - pass `max_days=H` to keep the eval honest.
    """
    ce = add_event_columns(clean_test)
    pp, _ = build_person_period(ce, hz["num"], hz["cat"], cat_dtypes=hz["cat_dtypes"])
    if max_days is not None:
        pp = pp[pp[AXIS] <= max_days]
    FEATS = hz["num"] + [AXIS] + hz["cat"]
    p = np.clip(hazard_fn(hz)(pp[FEATS]), 0.0, 1.0)
    return pd.DataFrame({AXIS: pp[AXIS].to_numpy(), "width": pp["width"].to_numpy(),
                         "y": pp["y"].to_numpy(), "p": p})


def snapshot_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot discrimination + calibration from `per_snapshot_scores` output:
    n, empirical vs predicted hazard, AUC, AP, Brier per days-until-arrival."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    rows = []
    for d, g in scores.groupby(AXIS):
        yv = g["y"].to_numpy(); pv = g["p"].to_numpy()
        rows.append({"days_until_arrival": int(d), "n": int(len(g)),
                     "emp_hazard": float(yv.mean()), "pred_hazard": float(pv.mean()),
                     "auc": float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan"),
                     "ap": float(average_precision_score(yv, pv)) if yv.sum() > 0 else float("nan"),
                     "brier": float(brier_score_loss(yv, pv))})
    return pd.DataFrame(rows).sort_values("days_until_arrival").reset_index(drop=True)


def walk_forward_per_snapshot(*, n_folds: int = 6, horizon_days: int = 14,
                              step_days: int = 14, seed: int = SEED) -> dict:
    """Decision-time walk-forward evaluated PER SNAPSHOT (time-resolved accuracy):
    fit per fold on train, score the fold's TEST bookings across ALL snapshots they
    traverse, pool, then report AUC/AP/Brier + empirical-vs-predicted hazard per
    days-until-arrival. Leak-free (each fold's test scored by that fold's model)."""
    from . import walkforward as wf, load_clean_reservations
    clean = wf.add_outcome_known_date(load_clean_reservations())
    folds = wf.make_folds(clean, n_folds=n_folds, horizon_days=horizon_days, step_days=step_days)
    parts = []
    for f in folds:
        tr, te = clean.iloc[f.train_idx], clean.iloc[f.test_idx]
        if len(tr) < 500 or len(te) < 50:
            continue
        hz = fit_hazard(tr, seed=seed)
        parts.append(per_snapshot_scores(hz, te, max_days=horizon_days))   # daily decision band only
    scores = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[AXIS, "width", "y", "p"])
    return {"per_snapshot": snapshot_metrics(scores), "scores": scores}


def _wf_hazard_folds(n_folds, horizon_days, step_days, seed):
    """Shared decision-time loop: yield (fold, test_frame_with_p, y) per fold, the
    hazard trained leak-free on that fold's train, scored at d = min(lead, H)."""
    from . import walkforward as wf, load_clean_reservations
    clean = wf.add_outcome_known_date(load_clean_reservations())
    folds = wf.make_folds(clean, n_folds=n_folds, horizon_days=horizon_days, step_days=step_days)
    for f in folds:
        tr, te = clean.iloc[f.train_idx], clean.iloc[f.test_idx]
        if len(tr) < 500 or len(te) < 50:
            continue
        hz = fit_hazard(tr, seed=seed)
        te = te.copy()
        arr = pd.to_datetime(te[ARRIVAL], utc=True); cre = pd.to_datetime(te["created"], utc=True)
        te["lead"] = (arr - cre) / pd.Timedelta(days=1)
        te[AXIS] = np.minimum(te["lead"], horizon_days).clip(lower=1)     # decision horizon d
        te["p"] = survival_cancel_proba(te, hazard_fn(hz), hz["num"], hz["cat"],
                                        hz["cat_dtypes"], snaps=hz.get("snap"))
        te["y"] = (wf.target_series(te).to_numpy() == 1).astype(int)
        yield f, te


def walk_forward_predict_hazard(*, n_folds: int = 8, horizon_days: int = 14,
                                step_days: int = 14, seed: int = SEED) -> pd.DataFrame:
    """Pooled PER-BOOKING hazard predictions on the decision-time folds (survival
    product at d = min(lead, H)). Returns [fold, y_true, y_prob] for reliability
    diagrams, Brier decomposition and matched comparisons."""
    parts = [pd.DataFrame({"fold": f.k, "y_true": te["y"].to_numpy(), "y_prob": te["p"].to_numpy()})
             for f, te in _wf_hazard_folds(n_folds, horizon_days, step_days, seed)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["fold", "y_true", "y_prob"])


def walk_forward_per_night(*, n_folds: int = 8, horizon_days: int = 14, step_days: int = 14,
                           seed: int = SEED, hotel_col: str | None = None) -> pd.DataFrame:
    """Pooled PER-ARRIVAL-NIGHT expected-vs-actual freed rooms across ALL decision-time
    folds (leak-free per fold) - the sample the aggregate overbooking calibration needs
    (one fold alone is ~14 nights = noise). Feed the result to `coverage_report`."""
    parts = []
    for f, te in _wf_hazard_folds(n_folds, horizon_days, step_days, seed):
        pn = per_night_table(te, te["p"].to_numpy(), hotel_col=hotel_col, label=te["y"].to_numpy())
        pn["fold"] = f.k
        parts.append(pn)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
