# ---------------------------------------------------------------------------
# Scoring: apply a trained model to (upcoming) bookings.
#
# Public entry points are score_upcoming() / score_reservations(). Both build
# the roster features, score through the one adapter cancel_proba() (works for
# the static pipelines AND the hazard model) and bucket each booking into
# low / uncertain / high risk.
#
# Feature engineering must match training: build_features() mirrors notebook 00
# §3.0, and the feature LISTS always come from Data/feature_roster.json. If you
# change a feature in 00, mirror it in build_features() and re-run 00.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final, Literal

import joblib
import numpy as np
import pandas as pd

from .data_loader import load_reservations
from .features import add_country_region
from .paths import data_dir, repo_root, tables_dir

logger = logging.getLogger(__name__)

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


# ---- Risk bucket fallbacks --------------------------------------------------
# serving_thresholds() derives the real cut points per model (high = cost-optimal
# threshold, low = observed base rate). These constants are only the last-resort
# fallbacks when neither validation predictions nor the cleaned-data metadata
# exist: LOW_THR ~ the historical cancel rate, HIGH_THR = the analytic threshold.
LOW_THR:  Final[float] = 0.20
HIGH_THR: Final[float] = analytic_threshold()


# ---- Model registry -------------------------------------------------------
# Every trainable model by name. `kind` decides how it is scored: "static" =
# sklearn Pipeline (predict_proba), "hazard" = survival artifact (src.hazard).
# All four stay registered so retrain()/notebooks can save them; SERVING is a
# separate decision made by resolve_model(): hazard is the default scorer,
# xgboost the fallback, logreg/histgb are comparison baselines only.
DEFAULT_MODEL:  Final[str] = "hazard"    # standard scoring model
FALLBACK_MODEL: Final[str] = "xgboost"   # used when the hazard artifact is absent

MODEL_REGISTRY: Final[dict[str, dict[str, str]]] = {
    "hazard":  {"kind": "hazard", "joblib": "08_hazard_model.joblib",
                "card": "reports/tables/08_hazard/model_card.json"},
    "xgboost": {"kind": "static", "joblib": "02_xgboost_model.joblib",
                "card": "reports/tables/02_xgboost/model_card.json"},
    "logreg":  {"kind": "static", "joblib": "01_logreg_model.joblib",
                "card": "reports/tables/01_logreg/model_card.json"},
    "histgb":  {"kind": "static", "joblib": "03_histgb_model.joblib",
                "card": "reports/tables/03_histgb/model_card.json"},
}


def _models_of_kind(kind: str) -> list[str]:
    """Registry names of a given kind ('static' / 'hazard')."""
    return [n for n, r in MODEL_REGISTRY.items() if r["kind"] == kind]


# =============================================================================
# Model loading
# =============================================================================

def list_available_models() -> list[str]:
    """Return the names of models whose joblib file actually exists on disk."""
    return [name for name, paths in MODEL_REGISTRY.items()
            if (data_dir() / paths["joblib"]).exists()]


def load_model(name: str):
    """Load a model artifact by registry name.

    Returns the RAW joblib object: a fitted sklearn Pipeline for static models
    (predict_proba), or the hazard artifact dict for the hazard model. Most callers
    should use `score_reservations()` / `cancel_proba()` instead, which handle the
    kind-specific scoring for you.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {list(MODEL_REGISTRY)}")
    p = data_dir() / MODEL_REGISTRY[name]["joblib"]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Train/copy the '{name}' artifact "
            f"({MODEL_REGISTRY[name]['joblib']}) into {data_dir()} first."
        )
    return joblib.load(p)


def load_model_card(name: str) -> dict:
    """Load the model_card.json saved by the model notebook."""
    p = repo_root() / MODEL_REGISTRY[name]["card"]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    with p.open("r") as fh:
        return json.load(fh)


# ---- Model selection: AP primary, calibration (Brier) gate ----------------
# One rule for "which static model is the fallback": rank by walk-forward AP
# (the right metric at ~20% prevalence), but only among models whose Brier is
# within `brier_tol` of the best — a sharp but miscalibrated ranker must never
# feed the overbooking decision. Reads the walk-forward block of the model card
# (the cards written by retrain() no longer carry `test_metrics`).
BRIER_TOL: Final[float] = 0.005


def _card_walkforward(name: str) -> dict:
    """{'ap': mean, 'brier': mean} from a model card's walk-forward block.
    Tolerates both card shapes (aggregate nested or flat). Missing -> Nones."""
    wf = load_model_card(name).get("walk_forward", {})
    agg = wf.get("aggregate", wf) if isinstance(wf, dict) else {}
    out = {}
    for m in ("ap", "brier"):
        cell = agg.get(m)
        out[m] = cell.get("mean") if isinstance(cell, dict) else None
    return out


def best_model(brier_tol: float = BRIER_TOL) -> str:
    """Best available STATIC model: highest walk-forward AP among those whose
    Brier is within `brier_tol` of the best. The hazard model is excluded on
    purpose — it is the default scorer and is judged on its own estimand."""
    avail = [n for n in list_available_models() if n in _models_of_kind("static")]
    if not avail:
        raise RuntimeError("no static model on disk — need 02_xgboost_model.joblib.")
    cards = {}
    for name in avail:
        try:
            m = _card_walkforward(name)
        except Exception:  # noqa: BLE001 — no card yet
            continue
        if m.get("ap") is not None:
            cards[name] = m
    if not cards:
        raise RuntimeError(
            "no static model card carries walk-forward metrics — retrain first "
            "(python main.py retrain --model xgboost)."
        )
    briers = [m["brier"] for m in cards.values() if m.get("brier") is not None]
    eligible = list(cards)
    if briers:
        best_brier = min(briers)
        eligible = [n for n, m in cards.items()
                    if (m.get("brier") if m.get("brier") is not None else float("inf"))
                    <= best_brier + brier_tol]
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


def brier_decomposition(y_true, y_prob, *, n_bins: int = 10) -> dict:
    """Murphy decomposition of the Brier score + Brier Skill Score.

        Brier = Reliability - Resolution + Uncertainty
          * Uncertainty = base_rate*(1-base_rate)  -> irreducible; DOMINATES at low
            prevalence, which is why a raw Brier of ~0.10 at a ~12% base rate is
            NOT bad - it is close to the best any model can do.
          * Reliability = mean squared gap between predicted and observed frequency
            per probability bin -> the CALIBRATION error (0 = perfectly calibrated).
          * Resolution  = how far the binned outcomes sit from the base rate -> SKILL
            (higher = better; a constant predictor has 0).
        BSS = 1 - Brier / Uncertainty  -> skill vs the base-rate ("climatology")
        predictor; > 0 means better than always guessing the base rate.

    Judge CALIBRATION by Reliability (and the reliability diagram), not raw Brier.
    """
    y = np.asarray(y_true, dtype=float); p = np.clip(np.asarray(y_prob, dtype=float), 0, 1)
    n = len(y); base = float(y.mean()); unc = base * (1 - base)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rel = res = 0.0
    for b in np.unique(idx):
        m = idx == b; nk = int(m.sum())
        if not nk:
            continue
        rel += nk * (p[m].mean() - y[m].mean()) ** 2
        res += nk * (y[m].mean() - base) ** 2
    rel /= n; res /= n
    brier = float(np.mean((p - y) ** 2))
    return {"brier": brier, "reliability": rel, "resolution": res, "uncertainty": unc,
            "bss": (1 - brier / unc) if unc > 0 else float("nan"), "base_rate": base}


def _val_predictions(name: str) -> "pd.DataFrame | None":
    """Out-of-time validation predictions (y_true, y_prob) for a model, or None.

    Reads Data/NN_<name>_predictions.parquet, written by training.retrain() from
    the pooled walk-forward predictions. Without the file, the threshold helpers
    fall back to the analytic values.
    """
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


def _clean_meta_base_rate() -> float | None:
    """Observed cancel rate from Data/reservations_clean_meta.json, or None."""
    p = data_dir() / "reservations_clean_meta.json"
    if not p.exists():
        return None
    try:
        share = json.loads(p.read_text()).get("positive_share")
        return float(share) if share is not None else None
    except Exception:  # noqa: BLE001 — unreadable metadata is not worth crashing for
        return None


def serving_thresholds(name: str, c_walk: float = COST_WALK,
                       c_empty: float = COST_EMPTY) -> tuple[float, float]:
    """(low, high) risk-bucket cut points for a model.

    high = the cost-optimal decision threshold (from persisted validation
    predictions, else the analytic value). low = the observed base rate
    (validation predictions -> cleaned-data metadata -> LOW_THR). Everything
    between "above-average risk" and "act on it" lands in 'uncertain'.
    """
    high = operating_threshold(name, c_walk, c_empty)
    v = _val_predictions(name)
    if v is not None:
        base = float(v["y_true"].mean())
    else:
        base = _clean_meta_base_rate() or LOW_THR
    return min(base, high), high


# =============================================================================
# Feature engineering (serving side)
# =============================================================================
# Feature LISTS always come from Data/feature_roster.json (written by 00 §11) —
# a hardcoded copy here is exactly the drift that once broke scoring. Loading is
# lazy so `import src` works before the artifact exists.

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
        Point-in-time "as-of" date for the DYNAMIC features (days_until_arrival,
        days_since_booking, pct_lead_time_elapsed, is_within_7d_of_arrival).
        * LIVE scoring: leave None -> wall-clock now (we score today's open bookings).
        * REPLAY / EVAL / TRAINING: you MUST pass the simulated scoring date S
          (e.g. a walk-forward origin). Leaving it at wall-clock there computes the
          dynamic features relative to "today" instead of S = leakage / nonsense.
        The STATIC features do not depend on `today`.

    Returns a NEW dataframe with all static + dynamic feature columns added;
    does not mutate. Drops nothing — the caller decides whether to drop NaNs.
    """
    out = df.copy()
    arrival   = pd.to_datetime(out["arrival"],   utc=True)
    departure = pd.to_datetime(out["departure"], utc=True)
    created   = pd.to_datetime(out["created"],   utc=True)
    # None => LIVE scoring (wall-clock). Replay/eval/training MUST pass S (see docstring).
    if today is None:
        today = pd.Timestamp.now("UTC").normalize()

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

    # Log twins (LINEAR family) — mirror 00 §3.0.i so the linear model is scoreable
    # on upcoming bookings (trees use the raw columns; the roster picks per family).
    for _b, _l in [("los_nights", "los_nights_log"), ("lead_time_days", "lead_time_days_log"),
                   ("gross_per_night", "gross_per_night_log"),
                   ("diff_gross_cancellation_fee", "diff_gross_cancellation_fee_log")]:
        out[_l] = np.log1p(pd.to_numeric(out[_b], errors="coerce").clip(lower=0))

    # has_children — mirrors 00 §3.0.g (roster feature).
    out["has_children"] = (
        (pd.to_numeric(out.get("children"), errors="coerce").fillna(0) > 0)
        | (_to_str_nz(out.get("childrenAges"), out.index).str.len() > 0)
    ).astype("Int64")

    # ratePlan_category: look up the TRAIN-fitted name->category map persisted in
    # the roster (00 §3.0.d/§11); unseen names -> "other". Recomputing the rare-name
    # collapse on scoring data would be a train/serve parity bug.
    from .features import load_feature_roster
    _rp_map = load_feature_roster().get("ratePlan_category_map", {})
    _rp_norm = (_to_str_nz(out.get("ratePlan_name"), out.index)
                .str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    _rp_cat = _rp_norm.map(_rp_map)
    out["ratePlan_category"] = _rp_cat.where(_rp_cat.notna(), "other").astype("object")
    # NOTE: property_name / channelCode / guaranteeType / unitGroup_name /
    # cancellationFee_name are raw pass-through columns and need no engineering.

    # ---- Dynamic features — POINT-IN-TIME relative to `today` --------------
    # Live = wall-clock now; replay/eval/training MUST pass the simulated date S
    # (else these leak). Consumed by the hazard model's day-axis at serving; NOT in
    # the static roster by default (see roster `dynamic_numeric`).
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
              low_thr: float = LOW_THR, high_thr: float = HIGH_THR) -> np.ndarray | str:
    """Map probability -> 'low' / 'uncertain' / 'high'.

    Scalars return a str, arrays a plain numpy array — deliberately NOT an
    indexed Series: assigning a fresh-index Series to a filtered frame once
    silently produced all-<NA> buckets. Pass thresholds from serving_thresholds().
    """
    if isinstance(prob, (int, float, np.floating)):
        if prob >= high_thr: return "high"
        if prob >= low_thr:  return "uncertain"
        return "low"
    p = np.asarray(prob)
    return np.where(p >= high_thr, "high",
           np.where(p >= low_thr,  "uncertain", "low"))


# ---- Which model do we score with? ----------------------------------------

def resolve_model(model_name: str | None = None) -> str:
    """Decide which model to score with.

    An explicit `model_name` wins (must be registered AND present on disk). With no
    name, use the DEFAULT (hazard) if its artifact exists, otherwise fall back to the
    static FALLBACK (xgboost). Raises if neither artifact is on disk.
    """
    avail = list_available_models()
    if model_name is not None:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"unknown model '{model_name}'. Known: {list(MODEL_REGISTRY)}")
        if model_name not in avail:
            raise FileNotFoundError(
                f"model '{model_name}' artifact not found on disk. Available: {avail or 'none'}."
            )
        return model_name
    if DEFAULT_MODEL in avail:
        return DEFAULT_MODEL
    if FALLBACK_MODEL in avail:
        logger.warning("hazard artifact missing — falling back to '%s'.", FALLBACK_MODEL)
        return FALLBACK_MODEL
    raise RuntimeError(
        "no scoring model available — need 08_hazard_model.joblib (standard) or "
        "02_xgboost_model.joblib (fallback) in the Data folder."
    )


# ---- One dispatch point: features -> per-booking cancel probability --------

def cancel_proba(model_name: str, feat: pd.DataFrame) -> np.ndarray:
    """Per-booking P(cancel at/before arrival) for a feature frame already produced
    by build_features(). This is the single place the kind-specific scoring lives:

      * static model  -> sklearn Pipeline.predict_proba[:, 1]
      * hazard model  -> survival product over the trained snapshot grid
                         (src.hazard.score_upcoming_hazard)

    Fails loud if build_features did not produce a column the chosen model needs,
    instead of a cryptic error deep inside the estimator.
    """
    kind = MODEL_REGISTRY[model_name]["kind"]

    if kind == "hazard":
        from . import hazard as hz_mod
        hz = hz_mod.load_hazard()
        needed = list(hz["num"]) + list(hz["cat"]) + [hz_mod.AXIS]
        missing = [c for c in needed if c not in feat.columns]
        if missing:
            raise KeyError(
                f"cancel_proba(hazard): build_features did not produce {missing}; "
                f"the hazard artifact expects these columns (incl. the day axis)."
            )
        return np.asarray(hz_mod.score_upcoming_hazard(hz, feat), dtype=float)

    # Static sklearn pipeline (xgboost in the MVP). The pipeline's ColumnTransformer
    # selects the columns it was trained on by name, so passing the roster superset
    # (numeric incl. log twins) is safe — the extras are dropped by the transformer.
    pipeline = load_model(model_name)
    num, cat = model_feature_lists()           # from the roster (single source of truth)
    needed = num + cat
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        raise KeyError(
            f"cancel_proba({model_name}): build_features did not produce roster "
            f"features {missing}. build_features must mirror 00_data_audit's engineering."
        )
    return pipeline.predict_proba(feat[needed])[:, 1]


# ---- Public scoring entry point --------------------------------------------

# Columns score_reservations / score_upcoming append to the feature frame.
_SCORE_COLS: Final[tuple[str, ...]] = (
    "cancel_proba", "pred_cancel", "cancel_threshold", "risk_bucket",
    "model_used", "scored_at",
)


def score_reservations(
    df: pd.DataFrame,
    model_name: str | None = None,
    *,
    threshold: float | None = None,
    today: pd.Timestamp | None = None,
    apply_bounds: bool = True,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Score an arbitrary set of (raw, PII-stripped) reservations — THE entry point.

    Builds features, optionally applies the light scoring bounds, then scores via
    `cancel_proba` (hazard by default, xgboost fallback — see resolve_model). Returns
    a NEW frame with the engineered features plus: cancel_proba, pred_cancel,
    cancel_threshold, risk_bucket, model_used, scored_at.

    Parameters
    ----------
    df : DataFrame
        Raw reservations (post PII strip), e.g. from load_reservations().
    model_name : str | None
        'hazard' or 'xgboost'. None -> the default (hazard) if its artifact exists.
    threshold : float | None
        Manual decision-threshold override (UI slider). None -> the model's
        cost-optimal validation threshold.
    today : pd.Timestamp | None
        As-of date for the DYNAMIC features. None -> wall-clock (live scoring). Pass
        the simulated scoring date for replay/eval (see build_features).
    apply_bounds : bool
        Drop rows whose essential features can't be computed (default True).
    save_as : str | None
        If given, also write the result to Data/<save_as> as parquet.
    """
    chosen = resolve_model(model_name)
    low_thr, high_thr = serving_thresholds(chosen)
    if threshold is not None:                       # manual override from the app
        high_thr = float(threshold)
        low_thr = min(low_thr, high_thr)

    feat = build_features(df, today=today)
    if apply_bounds:
        feat = apply_scoring_bounds(feat)

    if feat.empty:
        return pd.DataFrame(columns=list(feat.columns) + list(_SCORE_COLS))

    proba = cancel_proba(chosen, feat)
    feat = feat.copy()
    feat["cancel_proba"]     = proba
    feat["pred_cancel"]      = (proba >= high_thr).astype(int)
    feat["cancel_threshold"] = high_thr
    feat["risk_bucket"]      = bucketize(proba, low_thr, high_thr)  # ndarray -> position-safe
    feat["model_used"]       = chosen
    feat["scored_at"]        = pd.Timestamp.utcnow()

    if save_as:
        out_path = data_dir() / save_as
        feat.to_parquet(out_path, index=False)
        print(f"saved {len(feat):,} scored bookings → {out_path}")
    return feat


def score_upcoming(
    model_name: str | None = None,
    force_refresh: bool = False,
    save: bool = True,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score upcoming (future-arrival) bookings — thin wrapper over score_reservations.

    Loads the upcoming bookings from the reservations cache (force_refresh re-pulls
    BigQuery), scores them with the chosen model (hazard by default), and by default
    writes Data/scored_upcoming.parquet.

    Already-CANCELLED bookings are excluded before scoring — a cancelled booking has
    no bearing on future occupancy or cancellation risk, so it must never enter the
    scored set (the app's data layer filters again defensively).
    """
    df = load_reservations(force_refresh=force_refresh, upcoming_only=True)
    if "status" in df.columns:
        df = df[df["status"].astype("string") != "Canceled"].copy()
    return score_reservations(
        df, model_name=model_name, threshold=threshold,
        save_as="scored_upcoming.parquet" if save else None,
    )


