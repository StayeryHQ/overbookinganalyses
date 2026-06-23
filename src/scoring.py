# ---------------------------------------------------------------------------
# Scoring helpers — apply a trained model to upcoming arrivals.
#
# Why this lives in src/ and not in a notebook:
#   * It will be run on a schedule (e.g. daily), not interactively.
#   * It needs to be importable from the dashboards notebook (07) without
#     re-running a model notebook each time.
#
# What this module does:
#   * Loads any of the four trained joblib models (01-04) by name.
#   * Provides `score_upcoming()` which loads upcoming arrivals, applies the
#     same cleaning bounds + feature engineering each model notebook uses,
#     scores, and buckets bookings into low / uncertain / high risk.
#
# Feature engineering MUST mirror notebooks/00_data_audit.ipynb section 8c
# (the canonical recipe). It's duplicated here because the upcoming-arrivals
# scoring needs the same features computed on rows that did NOT go through
# the cleaning step. If you change a feature in 00, change it here too.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

import joblib
import numpy as np
import pandas as pd

from .data_loader import load_reservations
from .features import add_country_region
from .paths import data_dir, repo_root, tables_dir

# ---- Asymmetric overbooking costs -----------------------------------------
# The per-booking "treat as a cancellation" decision is cost-sensitive, NOT
# F1-balanced. Two error types, very different costs:
#   * False positive  -> we freed/oversold a room but the guest ARRIVES => walk a
#                        guest. Cost = COST_WALK.
#   * False negative  -> we kept the room but the booking CANCELS => empty room.
#                        Cost = COST_EMPTY.
# For CALIBRATED probabilities the Bayes-optimal decision is "flag cancel" when
#   p >= COST_WALK / (COST_WALK + COST_EMPTY)   (the analytic threshold below).
# Walking a guest is ~3.75x worse than an empty room, so the threshold is HIGH
# (~0.79): we act only when fairly sure. Defaults; the app passes its own costs.
COST_WALK:  Final[float] = 300.0   # cost of walking a guest (false positive)
COST_EMPTY: Final[float] = 80.0    # cost of an empty room   (false negative)


def analytic_threshold(c_walk: float = COST_WALK, c_empty: float = COST_EMPTY) -> float:
    """Bayes-optimal decision threshold for calibrated probabilities."""
    return c_walk / (c_walk + c_empty)


# ---- Risk bucket fallbacks (used only if val predictions are unavailable) --
# Normal operation derives the cut points per model from the validation
# predictions (cost-optimal threshold) via serving_thresholds(); these flat
# fallbacks only fire when those artifacts are missing.
LOW_THR:  Final[float] = analytic_threshold()
HIGH_THR: Final[float] = analytic_threshold()


# ---- Model registry -------------------------------------------------------
# Centralised so notebooks/dashboards refer to models by name, not by path.
# Add a new model here when you add a new model notebook.
#
# Lineup 2026-06-11 (see reports/open_decisions.md): logreg (01), xgboost (02),
# histgb (03). RandomForest and MLP are OUT of the lineup, so they are no longer
# registered here. Filenames follow the {NN}_{name}_model.joblib convention each
# model notebook saves with.
MODEL_REGISTRY: Final[dict[str, dict[str, str]]] = {
    "logreg":  {"joblib": "01_logreg_model.joblib",  "card": "reports/tables/01_logreg/model_card.json"},
    "xgboost": {"joblib": "02_xgboost_model.joblib", "card": "reports/tables/02_xgboost/model_card.json"},
    "histgb":  {"joblib": "03_histgb_model.joblib",  "card": "reports/tables/03_histgb/model_card.json"},
}


# =============================================================================
# Model loading
# =============================================================================

def list_available_models() -> list[str]:
    """Return the names of models whose joblib file actually exists on disk."""
    return [name for name, paths in MODEL_REGISTRY.items()
            if (data_dir() / paths["joblib"]).exists()]


def load_model(name: str):
    """Load a trained pipeline by registry name (e.g. 'xgb')."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {list(MODEL_REGISTRY)}")
    p = data_dir() / MODEL_REGISTRY[name]["joblib"]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run notebooks/0{list(MODEL_REGISTRY).index(name)+1}_*.ipynb first."
        )
    return joblib.load(p)


def load_model_card(name: str) -> dict:
    """Load the model_card.json saved by the model notebook."""
    p = repo_root() / MODEL_REGISTRY[name]["card"]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    with p.open("r") as fh:
        return json.load(fh)


def best_model_by_auc() -> str:
    """Return the name of the available model with the highest test AUC."""
    avail = list_available_models()
    if not avail:
        raise RuntimeError("no trained models on disk — run 01-03 first.")
    # NOTE: the model cards store the test AUC under the key "auc" (see the
    # `test_metrics` Series in notebooks 01-03), not "roc_auc".
    scored = [(name, load_model_card(name)["test_metrics"]["auc"])
              for name in avail]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ---- Model selection: AP primary, calibration (Brier) gate ----------------
# Single source of truth for "which static model do we serve": rank by test AP
# (right metric at ~18% prevalence), but only among models whose test Brier is
# within `brier_tol` of the best — so we never ship a sharp-but-miscalibrated
# ranker, since the probabilities feed the overbooking decision directly.
BRIER_TOL: Final[float] = 0.005


def best_model(brier_tol: float = BRIER_TOL) -> str:
    """Highest test AP among models whose test Brier is within `brier_tol` of best."""
    avail = list_available_models()
    if not avail:
        raise RuntimeError("no trained models on disk — run 01-03 first.")
    cards = {name: load_model_card(name)["test_metrics"] for name in avail}
    briers = [m["brier"] for m in cards.values() if "brier" in m]
    if briers:
        best_brier = min(briers)
        eligible = [n for n, m in cards.items()
                    if m.get("brier", float("inf")) <= best_brier + brier_tol]
    else:
        eligible = avail
    return max(eligible, key=lambda n: cards[n]["ap"])


# =============================================================================
# Cost-based operating point (shared definition — notebooks, scoring, app)
# =============================================================================
def cost_threshold_from_scores(y_true, y_prob,
                               c_walk: float = COST_WALK,
                               c_empty: float = COST_EMPTY) -> float:
    """Cost-minimising threshold from in-memory (y_true, y_prob): argmin over the
    score grid of `FP*c_walk + FN*c_empty`. The single shared definition (use it
    in the notebooks on validation scores AND in cost_optimal_threshold) so every
    model uses the same rule. Falls back to the analytic Bayes value if no scores.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if p.size == 0:
        return analytic_threshold(c_walk, c_empty)
    grid = np.unique(np.round(np.sort(p), 4))
    if grid.size == 0:
        return analytic_threshold(c_walk, c_empty)

    def total_cost(t: float) -> float:
        pred = p >= t
        fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
        return fp * c_walk + fn * c_empty

    return float(min(grid, key=total_cost))


def cost_at_threshold(y_true, y_prob, threshold: float,
                      c_walk: float = COST_WALK, c_empty: float = COST_EMPTY) -> dict:
    """Confusion counts + total cost at a given threshold (for op-point tables)."""
    y = np.asarray(y_true).astype(int); p = np.asarray(y_prob, dtype=float)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "total_cost": fp * c_walk + fn * c_empty}


def _val_predictions(name: str) -> "pd.DataFrame | None":
    """Load a model's persisted VALIDATION predictions (y_true, y_prob) or None."""
    fname = MODEL_REGISTRY[name]["joblib"].replace("_model.joblib", "_predictions.parquet")
    p = data_dir() / fname
    if not p.exists():
        return None
    try:
        d = pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None
    if "temporal_split" in d.columns:
        d = d[d["temporal_split"] == "val"]
    return d if {"y_true", "y_prob"}.issubset(d.columns) and len(d) else None


def cost_optimal_threshold(name: str, c_walk: float = COST_WALK,
                           c_empty: float = COST_EMPTY) -> float:
    """Cost-minimising threshold from a model's persisted validation predictions;
    analytic Bayes fallback when unavailable. Replaces the old F1-optimal point."""
    v = _val_predictions(name)
    if v is None:
        return analytic_threshold(c_walk, c_empty)
    return cost_threshold_from_scores(v["y_true"], v["y_prob"], c_walk, c_empty)


def operating_threshold(name: str, c_walk: float = COST_WALK,
                        c_empty: float = COST_EMPTY) -> float:
    """Default per-booking decision threshold = cost-optimal on validation."""
    return cost_optimal_threshold(name, c_walk, c_empty)


def serving_thresholds(name: str, c_walk: float = COST_WALK,
                       c_empty: float = COST_EMPTY) -> tuple[float, float]:
    """(low, high) risk-bucket cut points: high = cost-optimal val threshold,
    low = validation base rate (below-average-risk display band)."""
    high = operating_threshold(name, c_walk, c_empty)
    v = _val_predictions(name)
    base = float(v["y_true"].mean()) if v is not None else LOW_THR
    return min(base, high), high


# =============================================================================
# Feature engineering — MUST stay in sync with notebooks 01-04
# =============================================================================
# These are duplicated verbatim from the model notebooks. Any change in the
# notebook feature build MUST be mirrored here, otherwise scoring will use
# a different feature shape than training and quietly produce nonsense.

# --- Feature lists come from the ROSTER, not hardcoded here ----------------
# Single source of truth: Data/feature_roster.json, written by 00_data_audit
# §11 and loaded via src.features.load_feature_roster(). Hardcoding the lists
# here was the exact drift that broke 08 / score_upcoming, so we stopped.
#
# Loading is LAZY (inside functions / via module __getattr__) so that
# `import src` never requires the artifact to exist - it is only read when a
# scoring function actually runs.

def model_feature_lists() -> tuple[list[str], list[str]]:
    """(numeric, categorical) STATIC features for the trained pipelines,
    read from the roster artifact. Raises a clear error if 00 hasn't run."""
    from .features import load_feature_roster
    r = load_feature_roster()
    return list(r["numeric"]), list(r["categorical"])


def dynamic_features() -> list[str]:
    """Dynamic (scoring-time) feature names from the roster. Computed per-day in
    `build_features`; consumed by the hazard model (08), surfaced for the dashboard."""
    from .features import load_feature_roster
    return list(load_feature_roster().get("dynamic_numeric", []))


# Backward-compat: expose NUMERIC_FEATURES / CATEGORICAL_FEATURES / ALL_FEATURES
# / DYNAMIC_FEATURES as lazily-resolved module attributes (PEP 562), so existing
# `from src.scoring import NUMERIC_FEATURES` keeps working WITHOUT reading the
# JSON at import time.
def __getattr__(name: str):
    if name in ("NUMERIC_FEATURES", "CATEGORICAL_FEATURES", "ALL_FEATURES", "DYNAMIC_FEATURES"):
        num, cat = model_feature_lists()
        return {"NUMERIC_FEATURES": num, "CATEGORICAL_FEATURES": cat,
                "ALL_FEATURES": num + cat, "DYNAMIC_FEATURES": dynamic_features()}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _to_str_nz(s: pd.Series | None, index: pd.Index) -> pd.Series:
    """Coerce to string, treat NaN as empty string."""
    if s is None:
        return pd.Series([""] * len(index), index=index, dtype="string")
    return s.astype("string").fillna("")


def build_features(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build the model features from a raw (PII-stripped) reservations frame.

    Parameters
    ----------
    df : DataFrame
        Raw reservations (post PII strip).
    today : pd.Timestamp | None
        Reference "now" for the dynamic features. Defaults to
        `pd.Timestamp.utcnow().normalize()`. Pass an explicit timestamp when
        you want to replay historical scoring days (e.g. notebook 08).

    Returns a NEW dataframe with all static + dynamic feature columns added;
    does not mutate. Drops nothing — the caller decides whether to drop NaNs.
    """
    out = df.copy()
    arrival   = pd.to_datetime(out["arrival"],   utc=True)
    departure = pd.to_datetime(out["departure"], utc=True)
    created   = pd.to_datetime(out["created"],   utc=True)
    today     = today or pd.Timestamp.utcnow().normalize()

    # ---- Static features (mirror notebook 00 §3.0) -------------------------
    out["lead_time_days"]     = ((arrival.dt.normalize() - created.dt.normalize())
                                  / pd.Timedelta(days=1)).clip(lower=0)
    out["los_nights"]         = ((departure.dt.normalize() - arrival.dt.normalize())
                                  / pd.Timedelta(days=1))
    out["arrival_dow"]        = arrival.dt.dayofweek.astype("Int64")
    out["arrival_month"]      = arrival.dt.month.astype("Int64")
    out["is_weekend_arrival"] = (arrival.dt.dayofweek >= 5).astype("Int64")
    out["stay_bucket"]        = pd.cut(out["los_nights"], bins=[-1, 2, 6, 365],
                                       labels=["short", "mid", "long"]).astype("string")

    out["is_international"]   = (_to_str_nz(out.get("primaryGuest_address_countryCode"), out.index) != "DE").astype("Int64")
    # Structural region collapse of the high-cardinality country code. MUST use
    # the shared helper so scoring matches what 00 / the models train on.
    out = add_country_region(out)
    out["has_group"]          = ((_to_str_nz(out.get("groupName"), out.index).str.len() > 0) |
                                 (_to_str_nz(out.get("blockId"),  out.index).str.len() > 0)).astype("Int64")
    out["has_promo"]          = (_to_str_nz(out.get("promoCode"),     out.index).str.len() > 0).astype("Int64")
    out["has_corporate_code"] = (_to_str_nz(out.get("corporateCode"), out.index).str.len() > 0).astype("Int64")
    out["adults_n"]           = pd.to_numeric(out.get("adults"), errors="coerce").astype("Int64")

    gross = pd.to_numeric(out.get("totalGrossAmount_amount"), errors="coerce")
    out["gross_amount"]     = gross
    out["log_gross_amount"] = np.log1p(gross.clip(lower=0))
    out["gross_per_night"]  = gross / out["los_nights"].replace(0, np.nan)

    cf = pd.to_numeric(out.get("cancellationFee_fee_amount"), errors="coerce")
    out["diff_gross_cancellation_fee"] = gross - cf

    # has_children — mirrors 00 §3.0.g (roster feature).
    out["has_children"] = (
        (pd.to_numeric(out.get("children"), errors="coerce").fillna(0) > 0)
        | (_to_str_nz(out.get("childrenAges"), out.index).str.len() > 0)
    ).astype("Int64")

    # ratePlan_category — parity-safe: reproduce 00's FITTED vocabulary via the
    # persisted map (00 §3.0.d builds the buckets + collapses rare<50 -> "other"
    # on TRAIN, and §11 persists the resulting normalized-name -> category map).
    # Serving only normalizes the raw name the same way and looks it up; any name
    # not seen in training maps to "other" (the catch-all bucket). This avoids
    # recomputing the rare-collapse on scoring data (which would be a parity bug).
    from .features import load_feature_roster
    _rp_map = load_feature_roster().get("ratePlan_category_map", {})
    _rp_norm = (_to_str_nz(out.get("ratePlan_name"), out.index)
                .str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    _rp_cat = _rp_norm.map(_rp_map)
    out["ratePlan_category"] = _rp_cat.where(_rp_cat.notna(), "other").astype("object")
    # NOTE: property_name / channelCode / guaranteeType / unitGroup_name /
    # cancellationFee_name are raw pass-through columns and need no engineering.

    # ---- Dynamic features (surfaced for dashboard + hazard model) ----------
    days_to_arr = ((arrival.dt.normalize() - today) / pd.Timedelta(days=1))
    days_since  = ((today - created.dt.normalize()) / pd.Timedelta(days=1)).clip(lower=0)
    out["days_until_arrival"]      = days_to_arr.astype("float64")
    out["days_since_booking"]      = days_since.astype("float64")
    out["pct_lead_time_elapsed"]   = (days_since / out["lead_time_days"].replace(0, np.nan)) \
                                         .clip(0, 1).astype("float64")
    out["is_within_7d_of_arrival"] = (days_to_arr <= 7).astype("Int64")

    return out


def apply_scoring_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning at scoring time: drop rows where essential features are
    impossible to compute. Crucially does NOT filter on `status` — at scoring
    time most upcoming bookings are 'Confirmed' and that's the whole point.
    """
    keep = (
        (df["lead_time_days"].notna()) &
        (df["los_nights"] > 0) &
        (df["los_nights"] <= 90) &
        (df["log_gross_amount"].notna())
    )
    return df[keep].copy()


# =============================================================================
# Scoring
# =============================================================================

RiskBucket = Literal["low", "uncertain", "high"]


def bucketize(prob: float | np.ndarray,
              low_thr: float = LOW_THR, high_thr: float = HIGH_THR) -> pd.Series | str:
    """Map probability → 'low' / 'uncertain' / 'high' bucket.

    `low_thr` / `high_thr` default to the module fallbacks but are normally passed
    in from serving_thresholds() so buckets track the model's tuned operating point.
    """
    if isinstance(prob, (int, float, np.floating)):
        if prob >= high_thr: return "high"
        if prob >= low_thr:  return "uncertain"
        return "low"
    p = np.asarray(prob)
    out = np.where(p >= high_thr, "high",
          np.where(p >= low_thr,  "uncertain", "low"))
    return pd.Series(out, dtype="string")


def score_upcoming(
    model_name: str | None = None,
    force_refresh: bool = False,
    save: bool = True,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score upcoming arrivals with the chosen model.

    Parameters
    ----------
    model_name : str | None
        Registry name ('logreg', 'xgboost', 'histgb'). If None, pick via
        best_model() (highest test AP among well-calibrated models).
    force_refresh : bool
        Re-pull from BigQuery instead of using the parquet cache.
    save : bool
        If True (default), writes `Data/scored_upcoming.parquet` and returns it.
    threshold : float | None
        Manual decision threshold override (UI slider). None => the model's
        cost-optimal validation threshold (operating_threshold).

    Returns
    -------
    pd.DataFrame
        One row per upcoming booking with the engineered features, a
        `cancel_proba`, `pred_cancel`, `cancel_threshold`, and `risk_bucket`.
    """
    chosen = model_name or best_model()
    pipeline = load_model(chosen)
    low_thr, high_thr = serving_thresholds(chosen)
    if threshold is not None:                 # manual override from the app
        high_thr = float(threshold)
        low_thr = min(low_thr, high_thr)

    df = load_reservations(force_refresh=force_refresh, upcoming_only=True)
    feat = build_features(df)
    feat = apply_scoring_bounds(feat)

    if feat.empty:
        cols = list(feat.columns) + ["cancel_proba", "pred_cancel", "cancel_threshold",
                                     "risk_bucket", "model_used", "scored_at"]
        return pd.DataFrame(columns=cols)

    num, cat = model_feature_lists()           # from the roster (single source of truth)
    needed = num + cat
    # Fail loud: build_features must produce EVERY roster feature, otherwise the
    # pipeline silently sees a wrong shape. NB known parity gap (2026-06-18):
    # build_features does not yet engineer `ratePlan_category` (and a few other
    # 00-derived columns) - see TODO in build_features. This check surfaces it
    # clearly instead of a cryptic KeyError deep in sklearn.
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        raise KeyError(
            f"score_upcoming: build_features did not produce roster features {missing}. "
            f"build_features must mirror 00_data_audit's engineering for these columns."
        )
    X = feat[needed].copy()
    proba = pipeline.predict_proba(X)[:, 1]
    feat["cancel_proba"]     = proba
    feat["pred_cancel"]      = (proba >= high_thr).astype(int)
    feat["cancel_threshold"] = high_thr
    feat["risk_bucket"]      = bucketize(proba, low_thr, high_thr)
    feat["model_used"]       = chosen
    feat["scored_at"]        = pd.Timestamp.utcnow()

    if save:
        out_path = data_dir() / "scored_upcoming.parquet"
        feat.to_parquet(out_path, index=False)
        print(f"saved {len(feat):,} scored bookings → {out_path}")

    return feat


