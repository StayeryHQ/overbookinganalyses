# ---------------------------------------------------------------------------
# src/hazard.py
# Discrete-time cancellation HAZARD model (notebook 08), made serveable +
# retrainable from src so the app and the retrain API can use it like the
# static models.
#
# What it does
#   * expands bookings into a person-period grid (one row per day-before-arrival
#     snapshot) and learns h_d = P(cancel in window ending at d | survived, x);
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
SNAP_FINE: Final[list[int]] = list(range(1, 15))          # daily, served horizon
SNAP_COARSE: Final[list[int]] = [21, 30, 45, 60, 90]      # coarse train-only tail
SNAP: Final[list[int]] = sorted(SNAP_FINE + SNAP_COARSE)

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

    Mirrors nb08: event = status==1 AND cancel_days_before_arrival>0; event_d is
    the days-before-arrival timing; non-events are censored survivors. Keeps lead>0.
    """
    out = clean.copy()
    arr = pd.to_datetime(out["arrival"], utc=True, errors="coerce")
    cre = pd.to_datetime(out["created"], utc=True, errors="coerce")
    out["lead"] = (arr - cre) / pd.Timedelta(days=1)
    status = pd.to_numeric(out["status"], errors="coerce").fillna(0).astype(int)
    cdba = pd.to_numeric(out.get("cancel_days_before_arrival"), errors="coerce")
    is_event = status.eq(1) & cdba.gt(0)
    out["event_d"] = np.where(is_event, cdba, np.nan)
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
                          max_h: int = 14) -> np.ndarray:
    """Per-booking P(cancel before arrival) = 1 - Π_{u=1..D}(1-h_u).

    For each booking, build a FRESH forward grid u=1..min(D, max_h, lead) where D
    is the booking's `horizon_col` (days_until_arrival), score the hazard on it,
    and take the cumulative. `haz_fn(X)->h` lets tests pass a stub.
    """
    b = bookings.reset_index(drop=True).copy()
    b["_row"] = np.arange(len(b))
    D = np.minimum(np.floor(pd.to_numeric(b[horizon_col], errors="coerce").fillna(0)).astype(int), max_h)
    if "lead" in b.columns:
        D = np.minimum(D, np.floor(pd.to_numeric(b["lead"], errors="coerce").fillna(0)).astype(int))
    reps = D.clip(lower=0).to_numpy()
    if reps.sum() == 0:
        return np.zeros(len(b), dtype=float)
    # reset_index(drop=True): b.index.repeat duplicates index labels, which would make
    # the later .loc alignment return duplicates — use a clean RangeIndex instead.
    g = b.loc[b.index.repeat(reps)].reset_index(drop=True).copy()
    g[AXIS] = np.concatenate([np.arange(1, n + 1) for n in reps if n > 0]).astype("float64")
    for c in cat:
        g[c] = g[c].astype(cat_dtypes[c])                          # pin to train dtype
    for c in num + [AXIS]:
        g[c] = g[c].astype("float64")
    FEATS = num + [AXIS] + cat
    g["_h"] = np.clip(haz_fn(g[FEATS]), 0, 0.999999)
    g = g.sort_values(["_row", AXIS])
    g["_pcum"] = 1 - g.groupby("_row")["_h"].transform(lambda s: (1 - s).cumprod())
    # cumulative at the largest u per booking (sorted ascending -> last row of each group)
    last = g.groupby("_row")["_pcum"].last()                       # Series indexed by _row (0..n-1)
    out = np.zeros(len(b), dtype=float)
    out[last.index.to_numpy()] = last.to_numpy()
    return out


# =============================================================================
# Fit + calibrate (xgboost — kernel-validated)
# =============================================================================
def fit_hazard(clean_resolved: pd.DataFrame, *, val_frac: float = 0.15, seed: int = SEED) -> dict:
    """Fit the hazard model on RESOLVED bookings; HP-search + isotonic-calibrate
    on a temporally held-out (most-recent-by-created) validation slice.

    Returns a dict artifact: {model, iso, num, cat, cat_dtypes, snap, axis, hp, val_ap}.
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

    best = None
    for hp in HP_GRID:
        m = XGBClassifier(n_estimators=1200, tree_method="hist", enable_categorical=True,
                          eval_metric="aucpr", early_stopping_rounds=40,
                          objective="binary:logistic", n_jobs=-1, random_state=seed, **hp)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, hp, m)
    val_ap, hp, model = best
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(Xva)[:, 1], yva)
    return {"model": model, "iso": iso, "num": num, "cat": cat, "cat_dtypes": cat_dtypes,
            "snap": SNAP, "axis": AXIS, "hp": hp, "val_ap": float(val_ap),
            "n_train_pp": int(len(pp_tr))}


def hazard_fn(hz: dict):
    """Return a calibrated hazard callable haz(X)->h from a fitted artifact."""
    model, iso = hz["model"], hz["iso"]
    return lambda X: iso.predict(model.predict_proba(X)[:, 1])


def score_upcoming_hazard(hz: dict, bookings: pd.DataFrame) -> np.ndarray:
    """Per-booking P(cancel before arrival) for upcoming bookings (survival product)."""
    return survival_cancel_proba(bookings, hazard_fn(hz), hz["num"], hz["cat"], hz["cat_dtypes"])


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
def retrain_hazard(*, asof=None, persist: bool = True, seed: int = SEED) -> dict:
    """Fit the deployment hazard model on ALL data resolved by `asof` and persist."""
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
    return result
