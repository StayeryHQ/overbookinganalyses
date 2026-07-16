# dash_app/backend/model_ops.py
# Logic/read layer for the "Update & Retraining" page. Callbacks only orchestrate;
# everything here is importable and runnable WITHOUT Dash (the prerequisite for
# unattended automation later). Nothing is re-implemented: BigQuery access lives in
# src.data_loader, scoring in src.scoring, retraining in src.training/src.hazard,
# cost params in the shared cost-store. Rendering the page never touches BigQuery —
# model_status() only reads the tiny model-card JSON, so the tiles paint instantly.

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

import src
from src import scoring as sc
from src import training as tr

logger = logging.getLogger(__name__)

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
    """LOCAL-time display string ('Jul 13, 2026, 14:30 CEST') — one formatter
    for the whole app (src.fmt_ts_local); storage stays UTC."""
    return src.fmt_ts_local(value)


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
# THE data update — one strict BigQuery pull per table + immediate scoring
# =============================================================================
def update_all(progress: Progress = _noop, model_name: str | None = None,
               walk: float | None = None, empty: float | None = None,
               days: int = WINDOW_DAYS) -> dict:
    """Job entry point for the Update button (signature: progress first, so
    jobs.start can inject its reporter). Delegates to src.scoring.refresh_and_score
    — ONE BigQuery query per table, then scoring; NO silent cache fallback, a
    BigQuery failure raises and the job status shows it loudly.

    `walk`/`empty` (cost-store): when both are set, pred_cancel uses the analytic
    cost threshold for those costs. 0 € is a legitimate value, not "unset".
    """
    t0 = time.perf_counter()
    threshold = (sc.analytic_threshold(walk, empty)
                 if walk is not None and empty is not None else None)
    res = sc.refresh_and_score(model_name, days=days, threshold=threshold,
                               progress=progress)

    # Let the page backends see the fresh parquets on their next read.
    try:
        from dash_app.backend import data_access as da
        for fn in ("_reservations_cached", "_capacity_from_perf", "_property_code_to_name"):
            getattr(da, fn).cache_clear()
    except Exception as e:  # noqa: BLE001
        logger.warning("cache clear after update failed: %s", e)

    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    res["finished"] = _fmt_ts(res.get("finished_utc"))
    res["data_as_of"] = _fmt_ts(res.get("data_max_created"))
    res["model_label"] = model_label(res.get("model_used", ""))
    return res


def score_window_job(progress: Progress = _noop, model_name: str | None = None,
                     walk: float | None = None, empty: float | None = None,
                     days: int = WINDOW_DAYS) -> dict:
    """Occupancy 'Run scoring' job: pull ONLY the next `days` days of arrivals from
    BigQuery (a windowed SQL scan — NOT the full history) and score them.

    Hits BigQuery for a 14-day arrival window only, so it is cheap and always fresh;
    it deliberately does NOT touch the full-history reservations cache (that is the
    separate 'update historical data' action). Scores through the same src.scoring
    path as everything else, writes Data/scored_upcoming.parquet, and returns bucket
    counts for the green result card. Runs via the file-backed jobs runner (survives
    page changes) — NOT a Dash background callback (which left the old button stuck
    loading forever). Fails loudly if BigQuery is unavailable — no cache fallback.
    """
    t0 = time.perf_counter()
    progress(f"BigQuery: pulling the next {days} days of arrivals…", 0.15)
    df = src.load_reservations_upcoming_window(days=days, quiet=True)
    if "status" in df.columns:
        df = df[df["status"].astype("string") != "Canceled"].copy()

    threshold = (sc.analytic_threshold(walk, empty)
                 if walk is not None and empty is not None else None)
    progress(f"Scoring {len(df):,} bookings arriving in the next {days} days…", 0.55)
    scored = sc.score_reservations(df, model_name=model_name, threshold=threshold,
                                   save_as="scored_upcoming.parquet")

    try:  # let the page backends see the fresh scored parquet on their next read
        from dash_app.backend import data_access as da
        da._reservations_cached.cache_clear()
    except Exception as e:  # noqa: BLE001
        logger.warning("cache clear after scoring failed: %s", e)

    rb = scored.get("risk_bucket")
    buckets = ({b: int((rb == b).sum()) for b in ("high", "medium", "low")}
               if rb is not None else {"high": 0, "medium": 0, "low": 0})
    return {"scored_rows": int(len(scored)), "buckets": buckets, "days": int(days),
            "model_label": model_label(sc.resolve_model(model_name)),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "finished": _fmt_ts(pd.Timestamp.utcnow().isoformat())}


def update_history_job(progress: Progress = _noop) -> dict:
    """Cancellation-History 'update history' job: pull the FULL reservations history
    from BigQuery and rebuild the cleaned/labelled cache (src.build_clean_reservations,
    validated to match notebook 00). Refreshes the history views. Fails loudly on a
    BigQuery error — no cache fallback (the whole point is fresh history).
    """
    t0 = time.perf_counter()
    progress("BigQuery: pulling full reservations history…", 0.10)
    raw = src.load_reservations(force_refresh=True, quiet=True)   # also refreshes the raw cache
    if raw.empty:
        raise RuntimeError("BigQuery returned 0 reservations — refusing to rebuild the history.")

    progress(f"Cleaning + labelling {len(raw):,} rows…", 0.60)
    clean = src.build_clean_reservations(raw)
    progress("Writing the cleaned dataset…", 0.90)   # last cancel checkpoint before the write
    from src.data_loader import CLEAN_CACHE_FILE
    out_path = src.data_dir() / CLEAN_CACHE_FILE
    clean.to_parquet(out_path, index=False)

    try:  # let the Cancellation-History page see the fresh clean cache
        from dash_app.backend import cancellation_history as ch
        ch._clean.cache_clear()
        ch.property_list.cache_clear()
    except Exception as e:  # noqa: BLE001
        logger.warning("CH cache clear after history update failed: %s", e)

    arr = pd.to_datetime(clean["arrival"], utc=True, errors="coerce")
    return {"clean_rows": int(len(clean)), "raw_rows": int(len(raw)),
            "base_rate": round(float(pd.to_numeric(clean["status"]).mean()), 4),
            "span_start": str(arr.min().date()), "span_end": str(arr.max().date()),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "finished": _fmt_ts(pd.Timestamp.utcnow().isoformat())}


def shap_job(progress: Progress = _noop, model_name: str | None = None) -> dict:
    """Occupancy XAI 'Build explanations' job: compute global SHAP over the CURRENT
    14-day scored bookings for the served model (heavy → on demand). The single-booking
    waterfalls are computed live and need no pre-build.
    """
    from dash_app.backend import explain as ex
    model = model_name or sc.DEFAULT_MODEL
    progress(f"Computing SHAP over the scored bookings · {model_label(model)}…", 0.2)
    long = ex.compute_global_shap(model, refresh=True)
    progress("Done.", 1.0)
    return {"model": model, "model_label": model_label(model),
            "n_points": int(len(long)),
            "features": int(long["feature"].nunique()) if not long.empty else 0,
            "finished": _fmt_ts(pd.Timestamp.utcnow().isoformat())}


# =============================================================================
# Job wrappers — thin adapters with the (progress, *args) signature jobs.start
# expects. Keep ALL logic in the functions they call.
# =============================================================================
def retrain_job(progress: Progress, model_name: str, retune: bool = False) -> dict:
    return run_retrain(model_name, retune=retune, progress=progress)


def artifacts_job(progress: Progress, include_shap: bool = False) -> dict:
    """Build MISSING Model-Performance artifacts (eval; optionally SHAP)."""
    return ensure_all_eval(progress=progress, include_shap=include_shap)


def rebuild_eval_job(progress: Progress, model_name: str, all_models: bool = False) -> dict:
    """Force-rebuild the eval artifact(s) — the XAI page's 'Rebuild evaluation'."""
    from src import model_eval as me
    targets = list(me.EVAL_MODELS) if all_models else [model_name]
    done, errors = [], []
    for i, m in enumerate(targets):
        progress(f"Rebuilding evaluation · {model_label(m)}…", i / max(len(targets), 1))
        try:
            d = me.model_eval(m, refresh=True)
            done.append(f"{m} ({len(d):,} rows)")
        except Exception as e:  # noqa: BLE001 — collect, keep going, report loudly
            errors.append(f"{m}: {str(e)[:120]}")
    progress("Done.", 1.0)
    return {"rebuilt": done, "errors": errors}


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
    # Stage 1/2 — fit on all resolved data AND rebuild the Model-Performance eval artifact
    # (refresh_eval=True), so the performance page is never stale after a retrain.
    progress(f"Stage 1/2 · retraining '{model_label(model_name)}' ({mode}) + rebuilding "
             "evaluation. This can take a while…", 0.05)
    result = tr.retrain(model_name, mode=mode, asof=asof, persist=True, refresh_eval=True)

    # Stage 2/2 — rebuild this model's explanations (SHAP + iteration curve) so the XAI
    # section reflects the retrained model. Best-effort — never fail the retrain on it.
    progress("Stage 2/2 · rebuilding explanations (SHAP) for the updated model…", 0.55)
    try:
        from dash_app.backend import explain as ex
        ex.compute_global_shap(model_name, refresh=True)
        if model_name in ("xgboost", "histgb"):
            ex.iteration_curve(model_name, refresh=True)
    except Exception as e:  # noqa: BLE001 — never fail the retrain over explanations
        logger.warning("post-retrain SHAP rebuild failed for %s: %s", model_name, e)

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
                "high": 0, "medium": 0, "low": 0}
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
            "medium": int((rb == "medium").sum()) if rb is not None else 0,
            "low": int((rb == "low").sum()) if rb is not None else 0}


# Columns of the COMPACT export — what a revenue manager actually reads. The full
# export (all ~100 engineered columns) stays available for analysts.
_EXPORT_COLUMNS: tuple[str, ...] = (
    "property_name", "arrival", "departure", "los_nights", "channelCode",
    "ratePlan_name", "unitGroup_name", "totalGrossAmount_amount",
    "cancel_proba", "risk_bucket", "pred_cancel", "model_used", "scored_at",
)


def scored_export_frame(slim: bool = True) -> pd.DataFrame:
    """The scored set for download. `slim=True` (default) keeps only the columns
    a revenue manager reads; `slim=False` exports everything incl. features."""
    from dash_app.backend import data_access as da
    df = da.load_scored()
    if df.empty or not slim:
        return df
    cols = [c for c in _EXPORT_COLUMNS if c in df.columns]
    return df[cols].sort_values("cancel_proba", ascending=False)
