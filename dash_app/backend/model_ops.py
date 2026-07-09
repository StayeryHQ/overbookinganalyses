# dash_app/backend/model_ops.py
# Business-logic / read layer for the "Update & Retraining" page. Everything the page needs
# lives here as separable, testable functions so the callbacks only ORCHESTRATE — the actual
# data pull / scoring / retraining logic is importable and runnable WITHOUT Dash (that is the
# prerequisite for the later, un-attended automation the brief asks for).
#
# Nothing here is a re-implementation:
#   * BigQuery access  -> src.data_loader   (windowed fast path + full slow refresh)
#   * scoring          -> src.scoring       (the one cancel_proba adapter, all 4 models)
#   * retraining       -> src.training.retrain / src.hazard.retrain_hazard (dispatched)
#   * model registry   -> src.scoring.MODEL_REGISTRY
#   * cost params      -> the shared global cost-store (read via backend.model_performance)
#
# No BigQuery is ever triggered by simply rendering the page: model_status() only reads the
# model card (a tiny JSON), so the tiles paint instantly (progressive-loading requirement).

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

import src
from src import data_loader as dl
from src import scoring as sc
from src import training as tr

# A progress reporter: report(message, fraction in [0, 1]). The page adapts this to Dash's
# set_progress; the logic functions stay decoupled from the exact progress components.
Progress = Callable[[str, float], None]


def _noop(_msg: str, _frac: float) -> None:
    return None


# ---- Cadence policy (soft, per the brief — a hint, never a hard lock) -------
RETRAIN_INTERVAL_DAYS: int = 182          # ~6 months (normal cadence)
RETRAIN_NEW_LOCATION_DAYS: int = 61       # ~2 months (when a new location opens)
WINDOW_DAYS: int = 14                     # fast-path forward window (matches Occupancy)

# Operational scoring uses ONLY the served models (decision A in the concept doc):
# hazard (default) + xgboost (fallback). logreg/histgb stay comparison baselines (XAI page).
SERVING_MODELS: tuple[str, ...] = (sc.DEFAULT_MODEL, sc.FALLBACK_MODEL)

MODEL_LABELS: dict[str, str] = {
    "hazard": "Hazard (survival)", "xgboost": "XGBoost",
    "histgb": "HistGB", "logreg": "Logistic Regression",
}


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


# =============================================================================
# Small helpers
# =============================================================================
def _fmt_ts(value) -> str | None:
    """'Jul 03, 2026, 20:26' from an ISO string / datetime; None if unparseable."""
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%b %d, %Y, %H:%M")


def _days_since(value) -> int | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return int((pd.Timestamp.now("UTC") - ts) / pd.Timedelta(days=1))


def _artifact_present(model_name: str) -> bool:
    """Whether the model's joblib exists on disk (registry-driven, no guessing)."""
    reg = sc.MODEL_REGISTRY.get(model_name)
    if not reg:
        return False
    return (src.data_dir() / reg["joblib"]).exists()


def available_serving_models() -> list[str]:
    """Served models whose artifact is actually on disk, default (hazard) first."""
    avail = [m for m in SERVING_MODELS if _artifact_present(m)]
    return avail or list(SERVING_MODELS)   # still show the choices even if not trained yet


def scoring_model_options() -> list[dict]:
    """Dropdown data for the fast-path model selector (default = hazard)."""
    return [{"label": model_label(m), "value": m} for m in available_serving_models()]


# =============================================================================
# Info tiles + hyperparameters (read-only from the model card — instant)
# =============================================================================
def _load_card(model_name: str) -> dict | None:
    try:
        return sc.load_model_card(model_name)
    except Exception:  # noqa: BLE001 — no card yet
        return None


def model_status(model_name: str) -> dict:
    """Everything the info tiles need for `model_name`, read ONLY from the model card +
    registry (no BigQuery, no model load). Any value not present in a real artifact is
    returned as None so the UI can say "unavailable" rather than fabricate it.
    """
    reg = sc.MODEL_REGISTRY.get(model_name, {})
    present = _artifact_present(model_name)
    card = _load_card(model_name) or {}

    retrained_raw = card.get("retrained_at")
    roster_hash = card.get("roster_hash")
    # A booking-count is stored for static cards (n_train_deploy); the hazard card stores
    # person-periods — surface both honestly rather than conflating them.
    n_train = card.get("n_train_deploy")
    n_pp = card.get("n_train_person_period")

    if not present:
        status_label = "not trained"
    elif model_name == sc.DEFAULT_MODEL:
        status_label = "active · default scorer"
    elif model_name == sc.FALLBACK_MODEL:
        status_label = "active · fallback"
    else:
        status_label = "trained (comparison baseline)"

    version = "—"
    if retrained_raw:
        d = pd.to_datetime(retrained_raw, utc=True, errors="coerce")
        stamp = d.strftime("%Y-%m-%d") if not pd.isna(d) else "?"
        version = f"{stamp}" + (f" · feat {roster_hash[:6]}" if roster_hash else "")

    return {
        "model": model_name,
        "label": model_label(model_name),
        "kind": reg.get("kind"),
        "artifact_present": present,
        "status_label": status_label,
        "is_serving": model_name in SERVING_MODELS,
        "retrained_at": _fmt_ts(retrained_raw),
        "retrained_days_ago": _days_since(retrained_raw),
        "mode": card.get("mode"),
        "asof": card.get("asof"),
        "roster_hash": roster_hash,
        "version": version,
        "n_train_deploy": int(n_train) if n_train is not None else None,
        "n_train_person_period": int(n_pp) if n_pp is not None else None,
        "card_path": str(src.repo_root() / reg["card"]) if reg else None,
    }


def _fmt_hp_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def hyperparams_rows(model_name: str) -> list[dict]:
    """[{'param':..,'value':..}] of the model's CURRENT hyperparameters, from the card.
    Empty-safe: returns a single 'unavailable' row if no card / no hyperparams."""
    card = _load_card(model_name) or {}
    hp = card.get("hyperparams") or {}
    if not hp:
        return [{"param": "hyperparameters", "value": "unavailable (model not trained yet)"}]
    return [{"param": k, "value": _fmt_hp_value(v)} for k, v in hp.items()]


def latest_walkforward(model_name: str) -> dict:
    """Compact walk-forward metrics from the card (mean±std of auc/ap/brier/cost), for the
    NON-blocking metrics panel. Empty dict if the card has none. Never recomputes."""
    card = _load_card(model_name) or {}
    wf = card.get("walk_forward") or {}
    # static cards store the aggregate under 'walk_forward'; be tolerant of shapes.
    agg = wf.get("aggregate", wf) if isinstance(wf, dict) else {}
    out = {}
    for m in ("auc", "ap", "brier", "cost"):
        cell = agg.get(m) if isinstance(agg, dict) else None
        if isinstance(cell, dict) and "mean" in cell:
            out[m] = {"mean": cell.get("mean"), "std": cell.get("std")}
    # hazard cards store a person-period val_ap instead — surface it clearly labelled.
    if not out and card.get("val_ap") is not None:
        out["val_ap_person_period"] = {"mean": card.get("val_ap"), "std": None}
    return out


def cadence_hint(model_name: str) -> dict:
    """Soft cadence guidance (NOT a hard lock). Compares the last retrain against the ~6-month
    interval and returns {'level': 'ok'|'due'|'unknown', 'text': ...}."""
    days = None
    card = _load_card(model_name)
    if card:
        days = _days_since(card.get("retrained_at"))
    if days is None:
        return {"level": "unknown",
                "text": "No recorded training date — retraining will create the first "
                        "model card. Recommended cadence: about every 6 months (or every "
                        "~2 months when a new location opens)."}
    if days >= RETRAIN_INTERVAL_DAYS:
        return {"level": "due",
                "text": f"Last retrained {days} days ago — at or beyond the ~6-month "
                        "recommended interval. A refit is reasonable now."}
    return {"level": "ok",
            "text": f"Last retrained {days} days ago. Recommended cadence is about every "
                    "6 months (or every ~2 months when a new location opens), so retraining "
                    "is usually not needed yet — this is a hint, not a hard limit."}


# =============================================================================
# FAST PATH — pull the next 14 days from BigQuery + score immediately
# =============================================================================
def fast_score_next_14d(model_name: str | None = None, *, walk: float | None = None,
                        empty: float | None = None, days: int = WINDOW_DAYS,
                        progress: Progress = _noop) -> dict:
    """Time-critical path: pull ONLY bookings arriving in the next `days` days from BigQuery
    (small), score them with the chosen model via the shared adapter, and write the canonical
    Data/scored_upcoming.parquet (the same file the Occupancy page reads). Returns a summary.

    `walk`/`empty` come from the shared cost-store; when given, the decision threshold is the
    analytic cost-optimal point for those costs, so buckets reflect the same costs as the
    other pages. `progress(msg, frac)` drives the page's progress bar.
    """
    t0 = time.perf_counter()
    chosen = sc.resolve_model(model_name)     # validates registry + on-disk artifact
    progress(f"Pulling next {days} days of arrivals from BigQuery…", 0.15)

    df = dl.load_upcoming_window(days=days, force_refresh=True, quiet=True)
    # Never score an already-cancelled booking (defensive; the data layer filters again too).
    if "status" in df.columns:
        df = df[df["status"].astype("string") != "Canceled"].copy()

    if df.empty:
        progress("No upcoming arrivals in the window.", 1.0)
        return {"rows": 0, "high": 0, "uncertain": 0, "low": 0, "model_used": chosen,
                "scored_at": _fmt_ts(pd.Timestamp.utcnow()), "elapsed_s": round(time.perf_counter() - t0, 1),
                "window_days": days, "empty": True}

    threshold = sc.analytic_threshold(walk, empty) if (walk and empty) else None
    progress(f"Scoring {len(df):,} bookings with '{model_label(chosen)}'…", 0.55)
    scored = sc.score_reservations(df, model_name=chosen, threshold=threshold,
                                   save_as="scored_upcoming.parquet")

    progress("Writing scored results…", 0.9)
    # Let the Occupancy data layer see the fresh reservations rows on its next read.
    try:
        from dash_app.backend import data_access as da
        da._reservations_cached.cache_clear()
    except Exception:  # noqa: BLE001
        pass

    rb = scored.get("risk_bucket")
    out = {
        "rows": int(len(scored)),
        "high": int((rb == "high").sum()) if rb is not None else 0,
        "uncertain": int((rb == "uncertain").sum()) if rb is not None else 0,
        "low": int((rb == "low").sum()) if rb is not None else 0,
        "model_used": chosen,
        "threshold": float(threshold) if threshold is not None else None,
        "scored_at": _fmt_ts(pd.Timestamp.utcnow()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "window_days": days,
        "empty": False,
    }
    progress("Done.", 1.0)
    return out


# =============================================================================
# SLOW PATH — refresh the full history needed for the historical cancel rate
# =============================================================================
def slow_refresh_history(progress: Progress = _noop) -> dict:
    """Non-time-critical background refresh: re-pull the FULL reservations table and the
    property_performance_daily table (the data the historical cancellation rate is built on),
    rebuild their parquet caches, and clear the derived in-memory caches. Does NOT re-score —
    the fast path already scored the 14-day window (no redundant pull/score)."""
    t0 = time.perf_counter()
    progress("Refreshing full reservations history from BigQuery…", 0.2)
    resv = dl.load_reservations(force_refresh=True, quiet=True)

    progress("Refreshing property performance (occupancy + counts)…", 0.7)
    perf = dl.load_property_performance(force_refresh=True, quiet=True)

    progress("Rebuilding derived caches…", 0.95)
    try:
        from dash_app.backend import data_access as da
        for fn in ("_reservations_cached", "_capacity_from_perf", "_property_code_to_name"):
            getattr(da, fn).cache_clear()
    except Exception:  # noqa: BLE001
        pass

    progress("Done.", 1.0)
    return {"reservations_rows": int(len(resv)), "perf_rows": int(len(perf)),
            "refreshed_at": _fmt_ts(pd.Timestamp.utcnow()),
            "elapsed_s": round(time.perf_counter() - t0, 1)}


# =============================================================================
# RETRAIN — thin adapter over the existing, tested retrain logic
# =============================================================================
def run_retrain(model_name: str, *, retune: bool = False, asof: str | None = None,
                progress: Progress = _noop) -> dict:
    """Retrain ONE model for deployment. mode='refit' keeps the frozen card hyperparameters
    (default, per the brief); retune=True re-searches them. Delegates entirely to
    src.training.retrain (which dispatches hazard to src.hazard.retrain_hazard) — no modelling
    logic is duplicated here. Returns a compact, UI-friendly summary."""
    mode = "retune" if retune else "refit"
    progress(f"Retraining '{model_label(model_name)}' ({mode}) — fitting on all resolved "
             "data. This can take a while…", 0.1)

    # refresh_eval=True rebuilds the Model-Performance eval artifact for this model right
    # after the new model is persisted, so the comparison page never lags the deployed model.
    result = tr.retrain(model_name, mode=mode, asof=asof, persist=True, refresh_eval=True)

    # Also rebuild this model's explanations (SHAP beeswarm/importance + iteration curve) so
    # the XAI page reflects the retrained model. Best-effort — never fail the retrain on it.
    progress("Rebuilding explanations (SHAP) for the updated model…", 0.85)
    try:
        from dash_app.backend import explain as ex
        ex.compute_global_shap(model_name, refresh=True)
        if model_name in ("xgboost", "histgb"):
            ex.iteration_curve(model_name, refresh=True)
    except Exception:  # noqa: BLE001
        pass

    progress("Reading back the updated model card…", 0.9)
    status = model_status(model_name)
    agg = {}
    wf = result.get("walk_forward", {})
    if isinstance(wf, dict):
        agg = wf.get("aggregate", {})
    progress("Done.", 1.0)
    return {
        "model": model_name,
        "mode": result.get("mode", mode),
        "asof": result.get("asof"),
        "n_train_deploy": result.get("n_train_deploy"),
        "retrained_at": status.get("retrained_at"),
        "feature_change": result.get("feature_change"),
        "walk_forward_aggregate": agg,
        "hyperparams": result.get("hyperparams"),
        "eval_refreshed": result.get("eval_refreshed"),
    }


# =============================================================================
# Model-Performance artifacts — auto-warm (eval for all models; optional SHAP)
# =============================================================================
def eval_coverage() -> dict:
    """Which models already have a Model-Performance eval artifact on disk."""
    from src import model_eval as me
    have = [m for m in me.EVAL_MODELS if me.eval_available(m)]
    return {"have": have, "all": list(me.EVAL_MODELS),
            "complete": len(have) == len(me.EVAL_MODELS)}


def ensure_all_eval(progress: Progress = _noop, *, include_shap: bool = False) -> dict:
    """Build any MISSING Model-Performance eval artifacts (all four models), so the XAI page
    always has data — runs in the background, skips artifacts that already exist. With
    `include_shap=True` also builds the (slower) global SHAP for any model missing it.
    Never raises: per-model failures are collected, not fatal."""
    from src import model_eval as me
    models = list(me.EVAL_MODELS)
    n = len(models) * (2 if include_shap else 1)
    out: dict = {"built_eval": [], "have_eval": [], "built_shap": [], "errors": []}
    step = 0
    for m in models:
        step += 1
        progress(f"Evaluation · {model_label(m)}…", step / (n + 1))
        if me.eval_available(m):
            out["have_eval"].append(m)
        else:
            try:
                me.model_eval(m, refresh=False)      # builds only if missing
                out["built_eval"].append(m)
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"eval {m}: {str(e)[:60]}")
    if include_shap:
        from dash_app.backend import explain as ex
        for m in models:
            step += 1
            progress(f"Explanations · {model_label(m)}…", step / (n + 1))
            if ex.shap_available(m):
                continue
            try:
                ex.compute_global_shap(m, refresh=False)
                out["built_shap"].append(m)
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"shap {m}: {str(e)[:60]}")
    progress("Done.", 1.0)
    return out


# =============================================================================
# Scored set — directly retrievable on the page (table + export)
# =============================================================================
def scored_overview(limit: int = 1000) -> dict:
    """Summary + display rows of the current scored set (Data/scored_upcoming.parquet),
    highest cancel risk first. This is the 'scoring directly retrievable' surface."""
    from dash_app.backend import data_access as da
    df = da.load_scored()
    if df.empty:
        return {"n": 0, "rows": [], "scored_at": None, "model_used": None,
                "high": 0, "uncertain": 0, "low": 0}
    d = df.sort_values("cancel_proba", ascending=False)
    rb = d.get("risk_bucket")
    rows = []
    for r in d.head(limit).itertuples():
        arr = getattr(r, "arrival", None)
        rows.append({
            "property_name": getattr(r, "property_name", "—"),
            "arrival": pd.to_datetime(arr, utc=True, errors="coerce").strftime("%Y-%m-%d")
                       if arr is not None else "—",
            "cancel_pct": round(float(getattr(r, "cancel_proba", 0)) * 100, 1),
            "risk_bucket": getattr(r, "risk_bucket", "—"),
        })
    scored_at = _fmt_ts(d["scored_at"].iloc[0]) if "scored_at" in d.columns and len(d) else None
    model_used = d["model_used"].iloc[0] if "model_used" in d.columns and len(d) else None
    return {"n": int(len(df)), "rows": rows, "scored_at": scored_at, "model_used": model_used,
            "high": int((rb == "high").sum()) if rb is not None else 0,
            "uncertain": int((rb == "uncertain").sum()) if rb is not None else 0,
            "low": int((rb == "low").sum()) if rb is not None else 0}


def scored_export_frame() -> pd.DataFrame:
    """The full scored set for download (directly retrievable scores)."""
    from dash_app.backend import data_access as da
    return da.load_scored()
