# dash_app/backend/model_performance.py
# Read-only data + aggregation layer for the XAI / Model-Performance page. Reads ONLY the
# pre-computed leak-free eval artifacts (Data/model_eval_<model>.parquet, written by
# `python main.py eval`), never trains or queries BigQuery inline. Every metric is derived
# from ONE pooled, matched-estimand prediction set per model, so the four models — and the
# naive historical-average baseline — are always compared on the same footing.
#
# Baseline discipline (enforced here so no chart can get it wrong): the historical-average
# baseline is a CONSTANT predictor, meaningful for calibration / Brier / cost — NEVER for
# ROC-AUC (0.5 by construction). So roc_* functions carry NO baseline; the reliability,
# threshold-cost and KPI functions do.

from __future__ import annotations

import numpy as np
import pandas as pd

from src import model_eval as me
from src import scoring as sc

# Default asymmetric costs (walk a guest vs empty room) — the single shared definition.
DEFAULT_WALK: float = sc.COST_WALK      # 300.0
DEFAULT_EMPTY: float = sc.COST_EMPTY    # 80.0
GLOBAL_COST_KEY: str = "__global__"     # cost-store key for the page's global default

LOC_MIN_N: int = 200                    # min pooled bookings for a per-location metric


# ---------------------------------------------------------------------------
# Model + artifact availability
# ---------------------------------------------------------------------------
def registered_models() -> list[str]:
    """Every model the page can show, in display order."""
    return list(me.EVAL_MODELS)


def eval_status(model: str) -> dict:
    """{'available': bool, 'meta': provenance dict | None} for a model's eval artifact."""
    return {"available": me.eval_available(model), "meta": me.model_eval_meta(model)}


def load_eval(model: str) -> pd.DataFrame:
    """Pooled predictions for a model (empty frame if the artifact isn't built yet)."""
    if not me.eval_available(model):
        return pd.DataFrame(columns=list(me.EVAL_COLS))
    return me.model_eval(model)          # reads the cached parquet (no compute)


def _filtered(model: str, props: list[str] | None) -> pd.DataFrame:
    df = load_eval(model)
    if df.empty or not props:
        return df
    return df[df["property_name"].isin(props)].copy()


def location_options(model: str) -> list[str]:
    df = load_eval(model)
    if df.empty or "property_name" not in df.columns:
        return []
    return sorted(df["property_name"].dropna().astype(str).unique().tolist())


# ---------------------------------------------------------------------------
# Cost parameters — ONE global entry (GLOBAL_COST_KEY) shared by every page.
# Occupancy & Predictions is the primary entry point; Model Performance reads and
# writes the same entry (editable, kept in sync).
# ---------------------------------------------------------------------------
def read_cost_params(store: dict | None) -> tuple[float, float]:
    """(walk, empty) from the shared global cost entry, falling back to project
    defaults. Convenience wrapper over read_cost_full for the 2-tuple callers."""
    walk, empty, _high, _mult = read_cost_full(store)
    return walk, empty


def read_cost_full(store: dict | None) -> tuple[float, float, bool, float]:
    """(walk, empty, high_demand, multiplier) from the shared global cost entry.

    The single source of truth for costs across the whole app. Missing pieces fall
    back to the project defaults (walk/empty) and the default high-demand multiplier.
    """
    s = (store or {}).get(GLOBAL_COST_KEY) or {}
    walk = s.get("walk")
    empty = s.get("empty")
    mult = s.get("mult")
    walk = float(walk) if walk not in (None, "") else DEFAULT_WALK
    empty = float(empty) if empty not in (None, "") else DEFAULT_EMPTY
    mult = float(mult) if mult not in (None, "") else sc_default_multiplier()
    return walk, empty, bool(s.get("high", False)), mult


def sc_default_multiplier() -> float:
    """Default high-demand walk-cost multiplier (kept in src.overbooking)."""
    from src import DEFAULT_HIGH_DEMAND_MULTIPLIER
    return float(DEFAULT_HIGH_DEMAND_MULTIPLIER)


# ---------------------------------------------------------------------------
# 4.1 ROC — global + per location (NO baseline: constant predictor = 0.5)
# ---------------------------------------------------------------------------
def roc_global(model: str, props: list[str] | None) -> dict:
    from sklearn.metrics import roc_curve, roc_auc_score
    d = _filtered(model, props)
    if d.empty or d["y_true"].nunique() < 2:
        return {}
    y, p = d["y_true"].to_numpy(), d["y_prob"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y, p)), "n": int(len(d)),
            "base_rate": float(y.mean())}


def roc_by_location(model: str, props: list[str] | None, *, min_n: int = LOC_MIN_N) -> pd.DataFrame:
    """Per-location ROC-AUC on the pooled predictions. Locations with < min_n bookings or
    only one class present are dropped (never a fabricated number)."""
    from sklearn.metrics import roc_auc_score
    d = _filtered(model, props)
    if d.empty:
        return pd.DataFrame(columns=["property_name", "auc", "n", "pos"])
    rows = []
    for name, g in d.groupby("property_name"):
        y = g["y_true"].to_numpy()
        if len(g) < min_n or np.unique(y).size < 2:
            continue
        rows.append({"property_name": str(name), "auc": float(roc_auc_score(y, g["y_prob"].to_numpy())),
                     "n": int(len(g)), "pos": int(y.sum())})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4.2 Precision / Recall / F1 over the threshold + cost-optimal point + baseline
# ---------------------------------------------------------------------------
def pr_threshold(model: str, props: list[str] | None,
                 walk: float, empty: float, *, n_grid: int = 99) -> dict:
    d = _filtered(model, props)
    if d.empty or d["y_true"].nunique() < 2:
        return {}
    y = d["y_true"].to_numpy().astype(int)
    p = d["y_prob"].to_numpy()
    base = float(y.mean())
    grid = np.linspace(0.01, 0.99, n_grid)
    prec, rec, f1 = [], [], []
    for t in grid:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        pr = tp / (tp + fp) if (tp + fp) else np.nan
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        prec.append(pr); rec.append(rc)
        f1.append((2 * pr * rc / (pr + rc)) if (pr and rc and not np.isnan(pr)) else 0.0)
    # cost-optimal threshold from the shared definition, and its confusion/cost.
    t_cost = sc.cost_threshold_from_scores(y, p, walk, empty)
    cm = sc.cost_at_threshold(y, p, t_cost, walk, empty)
    # baseline (constant = base rate) operating point AT the same threshold.
    if base >= t_cost:                                  # predicts "cancel" for everyone
        b_prec, b_rec = base, 1.0
        b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec)
        b_cost = int(np.sum(y == 0)) * walk             # every negative is a false positive
    else:                                               # predicts "no cancel" for everyone
        b_prec, b_rec, b_f1 = 0.0, 0.0, 0.0
        b_cost = int(np.sum(y == 1)) * empty            # every positive is a false negative
    return {"thr": grid.tolist(), "precision": prec, "recall": rec, "f1": f1,
            "t_cost": float(t_cost), "cost": float(cm["total_cost"]),
            "cost_precision": float(cm["precision"]), "cost_recall": float(cm["recall"]),
            "base_rate": base, "walk": walk, "empty": empty,
            "baseline": {"precision": b_prec, "recall": b_rec, "f1": b_f1, "cost": float(b_cost)}}


# ---------------------------------------------------------------------------
# 4.3 Reliability diagram + Brier decomposition + baseline sanity
# ---------------------------------------------------------------------------
def reliability(model: str, props: list[str] | None, *, n_bins: int = 10) -> dict:
    d = _filtered(model, props)
    if d.empty or d["y_true"].nunique() < 2:
        return {}
    y = d["y_true"].to_numpy().astype(int)
    p = d["y_prob"].to_numpy()
    # Equal-width bins on [0,1]; keep only populated bins.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    bins = pd.DataFrame(rows)
    decomp = sc.brier_decomposition(y, p, n_bins=n_bins)
    # baseline: constant = base rate -> a single point at (base_rate, base_rate); its Brier
    # equals the "uncertainty" term (perfectly calibrated but zero resolution).
    base = float(y.mean())
    return {"bins": bins, "decomp": decomp, "base_rate": base, "n": int(len(d)),
            "mean_pred": float(p.mean())}


# ---------------------------------------------------------------------------
# 4.4 Train vs test (base view for all models; iteration curve added elsewhere)
# ---------------------------------------------------------------------------
def train_test(model: str) -> dict:
    fm = me.load_fold_metrics(model)
    if fm.empty:
        return {}
    agg = {}
    for col in ["train_auc", "test_auc", "train_brier", "test_brier", "train_ap", "test_ap"]:
        if col in fm.columns:
            agg[col] = {"mean": float(fm[col].mean()), "std": float(fm[col].std())}
    return {"per_fold": fm, "aggregate": agg, "n_folds": int(len(fm))}


# ---------------------------------------------------------------------------
# 4.9 KPI summary — model vs baseline, best/worst location (valid metric only)
# ---------------------------------------------------------------------------
def _brier_skill(y, p) -> float:
    """Brier Skill Score vs the constant base-rate predictor. >0 beats the baseline."""
    base = float(np.mean(y))
    unc = base * (1 - base)
    if unc <= 0:
        return float("nan")
    brier = float(np.mean((p - y) ** 2))
    return 1.0 - brier / unc


def kpis(model: str, props: list[str] | None, walk: float, empty: float,
         *, min_n: int = LOC_MIN_N) -> dict:
    """Headline KPIs for the selected model vs the naive baseline, plus the best/worst
    location by Brier Skill Score (a metric where the baseline comparison is valid — NOT
    AUC). Returns None-valued fields when the artifact is missing / below threshold."""
    from sklearn.metrics import roc_auc_score
    out = {"available": me.eval_available(model), "n": None, "base_rate": None,
           "auc": None, "brier": None, "bss": None, "cost_model": None, "cost_baseline": None,
           "best_loc": None, "best_bss": None, "worst_loc": None, "worst_bss": None}
    d = _filtered(model, props)
    if d.empty or d["y_true"].nunique() < 2:
        return out
    y = d["y_true"].to_numpy().astype(int)
    p = d["y_prob"].to_numpy()
    out["n"] = int(len(d))
    out["base_rate"] = float(y.mean())
    out["auc"] = float(roc_auc_score(y, p))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["bss"] = _brier_skill(y, p)
    # cost of the model at its cost-optimal threshold vs the baseline's best constant action.
    t_cost = sc.cost_threshold_from_scores(y, p, walk, empty)
    out["cost_model"] = float(sc.cost_at_threshold(y, p, t_cost, walk, empty)["total_cost"])
    out["cost_baseline"] = float(min(int(np.sum(y == 0)) * walk, int(np.sum(y == 1)) * empty))
    # best / worst location by BSS (baseline-valid), min-sample guarded.
    per = []
    for name, g in d.groupby("property_name"):
        yy = g["y_true"].to_numpy()
        if len(g) < min_n or np.unique(yy).size < 2:
            continue
        per.append((str(name), _brier_skill(yy, g["y_prob"].to_numpy())))
    if per:
        per.sort(key=lambda x: x[1], reverse=True)
        out["best_loc"], out["best_bss"] = per[0]
        out["worst_loc"], out["worst_bss"] = per[-1]
    return out
