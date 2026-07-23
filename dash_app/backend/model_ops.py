# dash_app/backend/model_ops.py
# Logic/read layer for the "Update & Retraining" page. Callbacks only orchestrate;
# everything here is importable and runnable WITHOUT Dash (the prerequisite for
# unattended automation later). Nothing is re-implemented: BigQuery access lives in
# src.data_loader, scoring in src.scoring, retraining in src.training/src.hazard,
# cost params in the shared cost-store. Rendering the page never touches BigQuery 
# model_status() only reads the tiny model-card JSON, so the tiles paint instantly.

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

import src
from src import scoring as sc

logger = logging.getLogger(__name__)

# A progress reporter: report(message, fraction in [0, 1]). The page adapts this to Dash's
# set_progress; the logic functions stay decoupled from the exact progress components.
Progress = Callable[[str, float], None]


def _noop(_msg: str, _frac: float) -> None:
    return None


# ---- Cadence policy (soft, per the brief  a hint, never a hard lock) -------
RETRAIN_INTERVAL_DAYS: int = 182          # ~6 months (normal cadence)
RETRAIN_NEW_LOCATION_DAYS: int = 61       # ~2 months (when a new location opens)
WINDOW_DAYS: int = 14                     # fast-path forward window (matches Occupancy)

# Served models (scoring + head-to-head performance): hazard (default) + xgboost
# (fallback) + histgb. ONE source of truth in src.scoring.SERVEABLE_MODELS, so the
# CLI and the app never drift. logreg stays a comparison-only baseline (not served).
SERVING_MODELS: tuple[str, ...] = sc.SERVEABLE_MODELS

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
    """LOCAL-time display string ('Jul 13, 2026, 14:30 CEST')  one formatter
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
# Info tiles + hyperparameters (read-only from the model card  instant)
# =============================================================================
def _load_card(model_name: str) -> dict | None:
    try:
        return sc.load_model_card(model_name)
    except Exception:  # noqa: BLE001  no card yet
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
    # person-periods  surface both honestly rather than conflating them.
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

    version = ""
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
    # hazard cards store a person-period val_ap instead  surface it clearly labelled.
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
                "text": "No recorded training date  retraining will create the first "
                        "model card. Recommended cadence: about every 6 months (or every "
                        "~2 months when a new location opens)."}
    if days >= RETRAIN_INTERVAL_DAYS:
        return {"level": "due",
                "text": f"Last retrained {days} days ago  at or beyond the ~6-month "
                        "recommended interval. A refit is reasonable now."}
    return {"level": "ok",
            "text": f"Last retrained {days} days ago. Recommended cadence is about every "
                    "6 months (or every ~2 months when a new location opens), so retraining "
                    "is usually not needed yet  this is a hint, not a hard limit."}


# =============================================================================
# THE data update  one strict BigQuery pull per table + immediate scoring
# =============================================================================
def score_window_job(progress: Progress = _noop, model_name: str | None = None,
                     walk: float | None = None, empty: float | None = None,
                     days: int = WINDOW_DAYS) -> dict:
    """Occupancy 'Run scoring' job  the 'rescore + refresh the occupancy view' action.

    Two BigQuery pulls, both cheap: the next `days` days of arrivals (a windowed SQL scan,
    NOT the full history) for scoring, AND the property-performance-daily table so the
    occupancy heatmap's capacity/occupancy is current. It deliberately does NOT touch the
    full-history reservations cache (that's the separate 'update historical data' action).
    Writes Data/scored_upcoming.parquet, refreshes the perf cache, and returns bucket
    counts for the green card. Runs on the file-backed jobs runner (survives page changes).
    Fails loudly if BigQuery is unavailable  no cache fallback.
    """
    t0 = time.perf_counter()
    progress(f"BigQuery: pulling the next {days} days of arrivals…", 0.15)
    df = src.load_reservations_upcoming_window(days=days, quiet=True)
    if "status" in df.columns:
        df = df[df["status"].astype("string") != "Canceled"].copy()

    # Refresh property performance so the occupancy graph (capacity / occupancy %) is current.
    progress("BigQuery: refreshing property performance (occupancy + capacity)…", 0.40)
    perf = src.load_property_performance(force_refresh=True, quiet=True)

    threshold = (sc.analytic_threshold(walk, empty)
                 if walk is not None and empty is not None else None)
    progress(f"Scoring {len(df):,} bookings arriving in the next {days} days…", 0.65)
    scored = sc.score_reservations(df, model_name=model_name, threshold=threshold,
                                   save_as="scored_upcoming.parquet")

    try:  # let the Occupancy backend see the fresh scored parquet + perf on its next read
        from dash_app.backend import data_access as da
        for fn in ("_reservations_cached", "_property_code_to_name", "_perf_daily"):
            getattr(da, fn).cache_clear()
    except Exception as e:  # noqa: BLE001
        logger.warning("cache clear after scoring failed: %s", e)

    rb = scored.get("risk_bucket")
    buckets = ({b: int((rb == b).sum()) for b in ("high", "medium", "low")}
               if rb is not None else {"high": 0, "medium": 0, "low": 0})
    return {"scored_rows": int(len(scored)), "buckets": buckets, "days": int(days),
            "perf_rows": int(len(perf)), "model_label": model_label(sc.resolve_model(model_name)),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "finished": _fmt_ts(pd.Timestamp.utcnow().isoformat())}


def training_rows_by_property() -> list[dict]:
    """[{property, rows}]  bookings per property in the cleaned TRAINING set (the clean
    cache). Descending by rows; empty list if the clean cache isn't built yet. Used for
    the retrain-page comparison and the scoring-page data-quality warning."""
    try:
        clean = src.load_clean_reservations()
    except Exception:  # noqa: BLE001  no clean cache yet
        return []
    if clean.empty or "property_name" not in clean.columns:
        return []
    vc = clean["property_name"].astype("string").value_counts()
    return [{"property": str(k), "rows": int(v)} for k, v in vc.items()]


def data_quality_flags(model_name: str, *, min_rows: int = 100,
                       stale_days: int = 182) -> dict:
    """Scoring-page data-quality signals: locations with < `min_rows` training bookings,
    and whether the model is older than `stale_days` (~6 months). Both are 'still works
    but may be less accurate' warnings, never hard blocks."""
    low = [r for r in training_rows_by_property() if r["rows"] < min_rows]
    st = model_status(model_name)
    days = st.get("retrained_days_ago")
    return {"low_locations": low, "min_rows": min_rows,
            "retrained_days_ago": days, "stale_days": stale_days,
            "is_stale": bool(days is not None and days > stale_days),
            "retrained_at": st.get("retrained_at")}


def clean_history_status() -> dict:
    """Freshness of the cleaned training history — the data the served model was trained on.

    Read STRAIGHT from the parquet: last-rebuilt time (file mtime), row count (parquet
    footer — cheap), and data span end (max of the `arrival` column). NOT from the meta
    sidecar, which only notebook 00 writes and would go stale after an in-app history
    rebuild. All values None-safe so the UI can say 'unavailable' rather than fabricate."""
    d = src.data_dir()
    clean = d / "reservations_clean.parquet"
    out = {"exists": clean.exists(), "rebuilt_at": None, "data_through": None, "rows": None}
    if not clean.exists():
        return out
    out["rebuilt_at"] = _fmt_ts(pd.to_datetime(clean.stat().st_mtime, unit="s", utc=True))
    try:
        import pyarrow.parquet as pq
        out["rows"] = int(pq.ParquetFile(clean).metadata.num_rows)      # footer only, cheap
        arr = pd.to_datetime(pd.read_parquet(clean, columns=["arrival"])["arrival"],
                             utc=True, errors="coerce").max()
        out["data_through"] = str(arr.date()) if pd.notna(arr) else None
    except Exception:  # noqa: BLE001
        pass
    return out


def update_history_job(progress: Progress = _noop) -> dict:
    """Cancellation-History 'update history' job: pull the FULL reservations history
    from BigQuery and rebuild the cleaned/labelled cache (src.build_clean_reservations,
    validated to match notebook 00). Refreshes the history views. Fails loudly on a
    BigQuery error  no cache fallback (the whole point is fresh history).
    """
    t0 = time.perf_counter()
    progress("BigQuery: pulling full reservations history…", 0.10)
    raw = src.load_reservations(force_refresh=True, quiet=True)   # also refreshes the raw cache
    if raw.empty:
        raise RuntimeError("BigQuery returned 0 reservations  refusing to rebuild the history.")

    progress(f"Cleaning + labelling {len(raw):,} rows…", 0.60)
    clean = src.build_clean_reservations(raw)
    progress("Writing the cleaned dataset…", 0.90)   # last cancel checkpoint before the write
    from src.data_loader import CLEAN_CACHE_FILE, write_clean_meta
    out_path = src.data_dir() / CLEAN_CACHE_FILE
    clean.to_parquet(out_path, index=False)
    write_clean_meta(clean)   # keep reservations_clean_meta.json fresh (model_meta / KPI read it)

    try:  # let the Cancellation-History page see the fresh clean cache
        from dash_app.backend import cancellation_history as ch
        ch._clean.cache_clear()
        ch.property_list.cache_clear()
        ch._noshow_prepared.cache_clear()   # no-show section reads the raw cache too
    except Exception as e:  # noqa: BLE001
        logger.warning("CH cache clear after history update failed: %s", e)

    arr = pd.to_datetime(clean["arrival"], utc=True, errors="coerce")
    return {"clean_rows": int(len(clean)), "raw_rows": int(len(raw)),
            "base_rate": round(float(pd.to_numeric(clean["status"]).mean()), 4),
            "span_start": str(arr.min().date()), "span_end": str(arr.max().date()),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "finished": _fmt_ts(pd.Timestamp.utcnow().isoformat())}


# =============================================================================
# Job wrappers  thin adapters with the (progress, *args) signature jobs.start
# expects. Keep ALL logic in the functions they call.
# =============================================================================
def rebuild_eval_job(progress: Progress, model_name: str, all_models: bool = False) -> dict:
    """Force-rebuild the eval artifact(s)  the XAI page's 'Rebuild evaluation'."""
    from src import model_eval as me
    targets = list(me.EVAL_MODELS) if all_models else [model_name]
    done, errors = [], []
    for i, m in enumerate(targets):
        progress(f"Rebuilding evaluation · {model_label(m)}…", i / max(len(targets), 1))
        try:
            d = me.model_eval(m, refresh=True)
            done.append(f"{m} ({len(d):,} rows)")
        except Exception as e:  # noqa: BLE001  collect, keep going, report loudly
            errors.append(f"{m}: {str(e)[:120]}")
    progress("Done.", 1.0)
    return {"rebuilt": done, "errors": errors}


# Retraining is CLI-only (`uv run python main.py retrain …` → src.training.retrain). The
# old in-app retrain_job/run_retrain wrappers were removed with the retrain button — the
# CLI calls src.training.retrain directly, so nothing was duplicated here.