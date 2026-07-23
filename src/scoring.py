# ---------------------------------------------------------------------------
# Scoring: apply a trained model to (upcoming) bookings.
#
# Public entry points are score_upcoming() / score_reservations(). Both build
# the roster features, score through the one adapter cancel_proba() (works for
# the static pipelines AND the hazard model) and bucket each booking into
# low / medium / high risk (thresholds: configs/risk_buckets.yaml).
#
# Feature engineering must match training: build_features() mirrors notebook 00
# §3.0, and the feature LISTS always come from Data/feature_roster.json. If you
# change a feature in 00, mirror it in build_features() and re-run 00.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from typing import Final, Literal

import joblib
import numpy as np
import pandas as pd

from .data_loader import load_reservations
from .features import add_country_region
from .paths import data_dir, repo_root

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


# NOTE on thresholds: ONE cost-based threshold now drives everything.
#   * The cost-optimal DECISION threshold (operating_threshold / analytic_threshold,
#     from the walk/empty costs) drives the yes/no column `pred_cancel` AND the
#     Low/Medium risk boundary in the UI.
#   * The fixed High cutoff (src.utils.HIGH_RISK_CUTOFF = 0.85) marks High risk,
#     always, independent of the threshold.
# There is no separate low_max/high_min risk scale any more.


# ---- Model registry -------------------------------------------------------
# Every trainable model by name. `kind` decides how it is scored: "static" =
# sklearn Pipeline (predict_proba), "hazard" = survival artifact (src.hazard).
# All four stay registered so retrain()/notebooks can save them; SERVING is a
# separate decision made by resolve_model(): hazard is the default scorer,
# xgboost the fallback. logreg stays a comparison-only baseline.
DEFAULT_MODEL:  Final[str] = "hazard"    # standard scoring model
FALLBACK_MODEL: Final[str] = "xgboost"   # used when the hazard artifact is absent

# Models the app offers for SERVING (scoring upcoming bookings) AND head-to-head
# performance comparison. User decision 2026-07: promote histgb to a full serving
# model alongside hazard + xgboost (previously served = {hazard, xgboost} only).
# Default first. logreg is deliberately excluded - it remains a baseline, not served.
# ONE source of truth: dash_app + main.py CLI derive their model choices from this.
SERVEABLE_MODELS: Final[tuple[str, ...]] = ("hazard", "xgboost", "histgb")

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
# within `brier_tol` of the best  a sharp but miscalibrated ranker must never
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


def _best_static_from_bakeoff(avail: list[str], brier_tol: float) -> "str | None":
    """Pick the best static from the FRESH matched bake-off (Data/bakeoff_predictions.parquet),
    where every model is scored on the IDENTICAL decision-time rows + label - so AP/Brier are
    directly comparable. Same rule as best_model (highest AP among Brier-eligible). Returns None
    if the bake-off is missing/stale or carries none of the available statics, so the caller can
    fall back to the per-card metrics."""
    try:
        from .training import load_bakeoff
        from sklearn.metrics import average_precision_score, brier_score_loss
        bake = load_bakeoff(require_fresh=True)       # StaleArtifact/FileNotFound => fall back
    except Exception:  # noqa: BLE001
        return None
    if bake is None or not len(bake) or "y_true" not in bake.columns:
        return None
    y = bake["y_true"].to_numpy()
    cand = [m for m in avail if f"p_{m}" in bake.columns]
    if not cand or len(set(y.tolist())) < 2:
        return None
    ap = {m: float(average_precision_score(y, bake[f"p_{m}"].to_numpy())) for m in cand}
    brier = {m: float(brier_score_loss(y, bake[f"p_{m}"].to_numpy())) for m in cand}
    best_brier = min(brier.values())
    eligible = [m for m in cand if brier[m] <= best_brier + brier_tol]
    return max(eligible, key=lambda m: ap[m])


def best_model(brier_tol: float = BRIER_TOL) -> str:
    """Best available STATIC model: highest walk-forward AP among those whose Brier is within
    `brier_tol` of the best. The hazard model is excluded on purpose  it is the default scorer
    and is judged on its own estimand.

    Prefers the MATCHED bake-off (all statics scored on identical rows) when a FRESH one exists,
    so the choice is not distorted by cards built at different times on different data snapshots.
    Falls back to the per-card walk-forward metrics if the bake-off is missing or stale."""
    avail = [n for n in list_available_models() if n in _models_of_kind("static")]
    if not avail:
        raise RuntimeError("no static model on disk  need 02_xgboost_model.joblib.")

    picked = _best_static_from_bakeoff(avail, brier_tol)
    if picked is not None:
        return picked

    cards = {}
    for name in avail:
        try:
            m = _card_walkforward(name)
        except Exception:  # noqa: BLE001  no card yet
            continue
        if m.get("ap") is not None:
            cards[name] = m
    if not cards:
        raise RuntimeError(
            "no static model card carries walk-forward metrics  retrain first "
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
# Cost-based operating point (shared definition  notebooks, scoring, app)
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




# =============================================================================
# Feature engineering (serving side)
# =============================================================================
# Feature LISTS always come from Data/feature_roster.json (written by 00 §11) 
# a hardcoded copy here is exactly the drift that once broke scoring. Loading is
# lazy so `import src` works before the artifact exists.

# NOTE: the former `scoring.model_feature_lists()` (raw-roster superset, no model arg)
# was removed in the 2026-07-22 feature-list unification. The single source of truth is
# now `src.features.model_feature_lists(model_name)` (family-correct); the raw-roster
# superset (presence guards / null audits) is `src.features.roster_features()`.


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


# OTA channel renames (mirror notebook 00 §3.0). Kept next to build_features so the
# ONE serving/training feature definition owns it.
_CHANNEL_RENAME: Final[dict[str, str]] = {
    "BookingCom": "Booking.com",
    "Expedia Affiliate Network": "Expedia",
}


def build_features(df: pd.DataFrame, today: pd.Timestamp | None = None,
                   *, rateplan_category_map: dict | None = None) -> pd.DataFrame:
    """Build the model features from a raw (PII-stripped) reservations frame.

    Parameters
    ----------
    df : DataFrame
        Raw reservations (post PII strip).
    rateplan_category_map : dict | None
        Explicit `normalized ratePlan_name -> category` map. None -> load it from the
        feature roster (the normal serving path). The explicit form lets the cleaner
        (build_clean_reservations) bootstrap features BEFORE a roster exists, so a
        fresh deploy is not stuck on a missing feature_roster.json.
    today : pd.Timestamp | None
        Point-in-time "as-of" date for the DYNAMIC features (days_until_arrival,
        days_since_booking, pct_lead_time_elapsed, is_within_7d_of_arrival).
        * LIVE scoring: leave None -> wall-clock now (we score today's open bookings).
        * REPLAY / EVAL / TRAINING: you MUST pass the simulated scoring date S
          (e.g. a walk-forward origin). Leaving it at wall-clock there computes the
          dynamic features relative to "today" instead of S = leakage / nonsense.
        The STATIC features do not depend on `today`.

    Returns a NEW dataframe with all static + dynamic feature columns added;
    does not mutate. Drops nothing  the caller decides whether to drop NaNs.
    """
    out = df.copy()

    # ---- Channel merge (mirror 00 §3.0.a)  fold `source` into ChannelManager rows,
    # then apply the OTA renames. Serving MUST match training here: otherwise a live
    # booking arrives as "ChannelManager" (a category the model never saw  training
    # remapped it to e.g. "Airbnb") and its channel signal is silently dropped.
    if "channelCode" in out.columns:
        chan = out["channelCode"].astype("string")
        if "source" in out.columns:
            src = out["source"].astype("string")
            m = (chan == "ChannelManager") & src.notna() & (src.str.len() > 0)
            chan = chan.mask(m, src)
        out["channelCode"] = chan.replace(_CHANNEL_RENAME)

    arrival   = pd.to_datetime(out["arrival"],   utc=True)
    departure = pd.to_datetime(out["departure"], utc=True)
    created   = pd.to_datetime(out["created"],   utc=True)
    # None => LIVE scoring (wall-clock). Replay/eval/training MUST pass S (see docstring).
    if today is None:
        today = pd.Timestamp.now("UTC").normalize()

    # ---- Static features (mirror notebook 00 §3.0) -------------------------
    # lead_time_days is the FRACTIONAL day gap (00 §3.0.b), clipped at 0  NOT rounded
    # to whole days. Training and serving must use the identical formula, or the model
    # sees systematically different lead values than it learnt on.
    out["lead_time_days"]     = ((arrival - created)
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

    # Log twins (LINEAR family)  mirror 00 §3.0.i so the linear model is scoreable
    # on upcoming bookings (trees use the raw columns; the roster picks per family).
    for _b, _l in [("los_nights", "los_nights_log"), ("lead_time_days", "lead_time_days_log"),
                   ("gross_per_night", "gross_per_night_log"),
                   ("diff_gross_cancellation_fee", "diff_gross_cancellation_fee_log")]:
        out[_l] = np.log1p(pd.to_numeric(out[_b], errors="coerce").clip(lower=0))

    # has_children  mirrors 00 §3.0.g (roster feature).
    out["has_children"] = (
        (pd.to_numeric(out.get("children"), errors="coerce").fillna(0) > 0)
        | (_to_str_nz(out.get("childrenAges"), out.index).str.len() > 0)
    ).astype("Int64")

    # ratePlan_category: look up the TRAIN-fitted name->category map persisted in
    # the roster (00 §3.0.d/§11); unseen names -> "other". Recomputing the rare-name
    # collapse on scoring data would be a train/serve parity bug.
    if rateplan_category_map is not None:
        _rp_map = rateplan_category_map
    else:
        from .features import load_feature_roster
        _rp_map = load_feature_roster().get("ratePlan_category_map", {})
    _rp_norm = (_to_str_nz(out.get("ratePlan_name"), out.index)
                .str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    _rp_cat = _rp_norm.map(_rp_map)
    out["ratePlan_category"] = _rp_cat.where(_rp_cat.notna(), "other").astype("object")
    # NOTE: property_name / channelCode / guaranteeType / unitGroup_name /
    # cancellationFee_name are raw pass-through columns and need no engineering.

    # ---- Dynamic features  POINT-IN-TIME relative to `today` --------------
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
    impossible to compute. Crucially does NOT filter on `status`  at scoring
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

RiskBucket = Literal["low", "medium", "high"]


def bucketize(prob: float | np.ndarray,
              threshold: float,
              high_cut: float | None = None) -> np.ndarray | str:
    """Map probability -> 'low' / 'medium' / 'high' using THE one cost-based rule
    (the same rule behind the UI's Low/Medium/High label  src.utils.risk_label_cost):
        p <  threshold          -> low
        threshold <= p < high_cut -> medium
        p >= high_cut           -> high
    `threshold` is the cost-optimal decision threshold; `high_cut` defaults to the
    fixed High cutoff (src.utils.HIGH_RISK_CUTOFF = 0.85).

    Scalars return a str, arrays a plain numpy array  deliberately NOT an
    indexed Series: assigning a fresh-index Series to a filtered frame once
    silently produced all-<NA> buckets.
    """
    from .utils import HIGH_RISK_CUTOFF
    thr = float(threshold)
    hi = float(HIGH_RISK_CUTOFF if high_cut is None else high_cut)
    if isinstance(prob, (int, float, np.floating)):
        if prob >= hi: return "high"
        if prob >= thr: return "medium"
        return "low"
    p = np.asarray(prob)
    return np.where(p >= hi, "high", np.where(p >= thr, "medium", "low"))


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
        logger.warning("hazard artifact missing  falling back to '%s'.", FALLBACK_MODEL)
        return FALLBACK_MODEL
    raise RuntimeError(
        "no scoring model available  need 08_hazard_model.joblib (standard) or "
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

    # Static sklearn pipeline. Use this model's FAMILY-CORRECT columns - the ONE source
    # of truth (src.features.model_feature_lists), exactly what the pipeline was fit on -
    # and fail loud if build_features did not produce one of them.
    from .features import model_feature_lists
    pipeline = load_model(model_name)
    num, cat = model_feature_lists(model_name)
    needed = num + cat
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        raise KeyError(
            f"cancel_proba({model_name}): build_features did not produce model "
            f"features {missing}. build_features must mirror 00_data_audit's engineering."
        )
    p = pipeline.predict_proba(feat[needed])[:, 1]
    return _apply_recalibration(model_name, p)


def _apply_recalibration(name: str, p: np.ndarray) -> np.ndarray:
    """Decision-time recalibration for STATIC models, if the artifact exists.

    The pipeline is isotonic-calibrated on the RESOLVED population (base ~20%),
    but consumed on the DECISION-TIME population ("still open at the decision
    date", base ~12%)  survivorship selection makes it overpredict there ~2x.
    training.retrain() fits a second isotonic map on the pooled decision-time
    walk-forward predictions (Data/NN_<name>_calibration.joblib); applying it
    here puts the served probabilities on the population they are used for.
    Monotone map -> ranking is preserved. Without the artifact: raw pass-through.
    """
    path = data_dir() / MODEL_REGISTRY[name]["joblib"].replace("_model.joblib",
                                                               "_calibration.joblib")
    if not path.exists():
        return p
    try:
        iso = joblib.load(path)
        return np.asarray(iso.predict(p), dtype=float)
    except Exception as e:  # noqa: BLE001  a broken map must not break scoring
        logger.warning("recalibration map for %s unreadable (%s); serving raw probs", name, e)
        return p


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
    """Score an arbitrary set of (raw, PII-stripped) reservations  THE entry point.

    Builds features, optionally applies the light scoring bounds, then scores via
    `cancel_proba` (hazard by default, xgboost fallback  see resolve_model). Returns
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
    # ONE cost-based rule: the cost-optimal decision threshold drives BOTH the yes/no
    # column pred_cancel AND the Low/Medium risk boundary; the fixed 0.85 cutoff
    # (src.utils.HIGH_RISK_CUTOFF) marks High. A manual `threshold` overrides the
    # cost-optimal value (e.g. from the app's cost inputs).
    decision_thr = (float(threshold) if threshold is not None
                    else operating_threshold(chosen))

    feat = build_features(df, today=today)
    if apply_bounds:
        feat = apply_scoring_bounds(feat)

    if feat.empty:
        return pd.DataFrame(columns=list(feat.columns) + list(_SCORE_COLS))

    proba = cancel_proba(chosen, feat)
    feat = feat.copy()
    feat["cancel_proba"]     = proba
    feat["pred_cancel"]      = (proba >= decision_thr).astype(int)
    feat["cancel_threshold"] = decision_thr
    feat["risk_bucket"]      = bucketize(proba, decision_thr)  # ndarray -> position-safe
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
    """Score upcoming (future-arrival) bookings  thin wrapper over score_reservations.

    Loads the upcoming bookings from the reservations cache (force_refresh re-pulls
    BigQuery), scores them with the chosen model (hazard by default), and by default
    writes Data/scored_upcoming.parquet.

    Already-CANCELLED bookings are excluded before scoring  a cancelled booking has
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


def refresh_and_score(model_name: str | None = None, *, days: int = 14,
                      threshold: float | None = None, progress=None) -> dict:
    """THE combined data update: one strict BigQuery pull per table, then score
    the next `days` days from the fresh data.

    Replaces the old fast(window-query)/slow(full-refresh) split  the full pull
    already contains the upcoming bookings, so ONE query per table serves both
    the history views and the scoring. There is deliberately NO cache fallback:
    if BigQuery fails, this raises and the data is explicitly NOT fresh.

    `progress(msg, frac)` is optional (drives job progress bars). Returns a
    summary incl. `data_max_created` (how fresh the pulled data actually is).
    """
    from .data_loader import load_property_performance, load_reservations

    def _p(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    chosen = resolve_model(model_name)
    _p("BigQuery: pulling full reservations history…", 0.05)
    resv = load_reservations(force_refresh=True, quiet=True)
    if resv.empty:
        raise RuntimeError("BigQuery returned 0 reservations  refusing to overwrite the cache.")

    _p("BigQuery: pulling property performance…", 0.45)
    perf = load_property_performance(force_refresh=True, quiet=True)

    _p(f"Scoring the next {days} days with '{chosen}'…", 0.60)
    start = pd.Timestamp.now("UTC").normalize()
    arr = pd.to_datetime(resv["arrival"], utc=True, errors="coerce")
    window = resv[(arr >= start) & (arr < start + pd.Timedelta(days=days))].copy()
    if "status" in window.columns:
        window = window[window["status"].astype("string") != "Canceled"].copy()
    scored = score_reservations(window, model_name=chosen, threshold=threshold,
                                save_as="scored_upcoming.parquet")

    _p("Done.", 1.0)
    created = pd.to_datetime(resv.get("created"), utc=True, errors="coerce")
    rb = scored.get("risk_bucket")
    return {
        "model_used": chosen,
        "reservations_rows": int(len(resv)),
        "perf_rows": int(len(perf)),
        "scored_rows": int(len(scored)),
        "buckets": ({b: int((rb == b).sum()) for b in ("high", "medium", "low")}
                    if rb is not None else {"high": 0, "medium": 0, "low": 0}),
        "threshold": float(threshold) if threshold is not None else None,
        "data_max_created": str(created.max()) if created.notna().any() else None,
        "finished_utc": pd.Timestamp.utcnow().isoformat(),
    }


