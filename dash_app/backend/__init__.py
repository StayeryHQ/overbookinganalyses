# dash_app/backend/__init__.py
# ---------------------------------------------------------------------------
# Backend facade: ONE interface for every page (REAL-only as of v11; the
# synthetic "dummy" backend was removed).
#
# Pages call ONLY functions from this module; the real implementation
# (`backend.real` -> `src.scoring` / `src.hazard`) returns EXACTLY the canonical
# schema (schema.COLUMNS), so pages never touch raw src/BigQuery columns. Model
# selection is funnelled through set_model()/get_model(); swapping the model is a
# backend-only change, never a page change.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import pandas as pd

# Re-export derive + schema so pages can do `from dash_app.backend import schema`.
from . import derive, schema  # noqa: F401
from . import schema as S
# Central config (paths, default model, horizon).
from .. import config as CFG

# Where the refresh-status JSON lives (just the last-refresh timestamp).
_DATA_DIR = CFG.DATA_DIR
_SNAPSHOT_JSON = _DATA_DIR / "refresh_status.json"

# How far ahead (days) the upcoming-arrivals window spans.
GENERATION_HORIZON_DAYS = CFG.GENERATION_HORIZON_DAYS

# Runtime model selection (module global so a dropdown choice persists across
# callbacks in one process). None => auto-pick the served model.
_MODEL_OVERRIDE: str | None = CFG.DEFAULT_MODEL


# =============================================================================
# Mode + model selection
# =============================================================================
def mode() -> str:
    """Active backend mode. The dummy backend was removed (v11) — always 'real'."""
    return "real"


def get_model() -> str | None:
    """Return the currently selected model name (None = auto-pick best model)."""
    return _MODEL_OVERRIDE


def set_model(name: str | None) -> None:
    """Select the model the real backend scores with (None = auto-pick)."""
    global _MODEL_OVERRIDE
    _MODEL_OVERRIDE = name


def available_models() -> list[str]:
    """List model registry names that have a trained joblib on disk.

    Falls back to an EMPTY list (not an error) if src can't be imported or no
    models exist — the UI then shows only the 'auto' option.
    """
    try:
        # Lazy import so listing models never drags in heavy ML deps at startup.
        from .real import _import_src
        src = _import_src()
        return list(src.list_available_models())
    except Exception:  # noqa: BLE001 — any failure => no models available
        return []


def retrain_models(mode_: str = "refit") -> dict:
    """Retrain the SERVED models via src and clear caches (the app's retrain hook).

    Retrains the optimal pair from src.training.select_models(): the best static
    model (mode_ = 'refit' frozen HP, or 'retune' re-search) AND the hazard model
    (the primary per-night engine). Returns a per-model summary with the honest
    walk-forward metrics. Heavy (minutes) — the page runs it on an explicit click.
    """
    from .real import _import_src
    _import_src()                                   # ensure repo root on sys.path
    from src import training as T, hazard as HZ     # noqa: E402
    out: dict = {}
    sel = T.select_models()
    if sel.get("static"):
        out["static"] = T.retrain(sel["static"], mode=mode_)
    out["hazard"] = HZ.retrain_hazard()
    # Invalidate cached scored bookings + model-perf artifacts so pages re-read.
    try:
        from . import real, model_perf
        if hasattr(real.get_scored_bookings, "cache_clear"):
            real.get_scored_bookings.cache_clear()
        for fn in (model_perf._load_card, model_perf._load_test_predictions,
                   model_perf._load_importance):
            fn.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    return out


def serving_bounds(model_name: str | None = None) -> tuple[float, float]:
    """(low, high) risk cut points for a model, from the training artifacts.

      * high = the COST-OPTIMAL threshold on validation (minimises FP*COST_WALK +
               FN*COST_EMPTY). This is the default decision point the UI slider
               starts at; analysts can override it.
      * low  = the validation base rate (a booking below the average cancel rate
               is by definition below-average risk) — only the display band.

    Falls back to the analytic Bayes threshold (COST_WALK/(COST_WALK+COST_EMPTY))
    for BOTH bounds when src/models are unavailable (e.g. no joblib on disk).
    """
    try:
        from .real import _import_src
        src = _import_src()
        name = model_name or src.best_model()
        low, high = src.serving_thresholds(name)
        return float(low), float(high)
    except Exception:  # noqa: BLE001 — no model/src => analytic fallback
        try:
            from .real import _import_src
            a = float(_import_src().analytic_threshold())
        except Exception:  # noqa: BLE001
            a = 0.79
        return a, a


def default_threshold(model_name: str | None = None) -> float:
    """The slider's default decision threshold = the model's cost-optimal value."""
    return serving_bounds(model_name)[1]


# =============================================================================
# Location helpers (read from configs/locations.yaml via locations.py)
# =============================================================================
@lru_cache(maxsize=1)
def units_by_hotel() -> dict[str, int]:
    """{propertyId(==hotel_code): units}.

    REAL: from property_performance_daily via property_universe() — this REPLACES
    configs/locations.yaml and picks up NEW propertyIds automatically (units =
    latest houseCount). Falls back to configs/locations.yaml when the table is
    unavailable (no creds / offline).
    """
    from . import occupancy
    uni = occupancy.units_from_universe()                   # real table or {}
    if uni:
        return uni
    from .locations import _load_locations                      # YAML fallback
    return {hc: units for hc, _city, units in _load_locations()}


@lru_cache(maxsize=1)
def city_by_hotel() -> dict[str, str]:
    """{hotel_code: city}."""
    from .locations import _load_locations
    return {hc: city for hc, city, _units in _load_locations()}


@lru_cache(maxsize=1)
def hotel_labels() -> dict[str, str]:
    """{hotel_code: display label}. Unique city -> just the city, else 'City (CODE)'."""
    cities = city_by_hotel()
    # Count how many properties share each city so we know when to disambiguate.
    counts: dict[str, int] = {}
    for city in cities.values():
        counts[city] = counts.get(city, 0) + 1
    return {hc: (city if counts[city] == 1 else f"{city} ({hc})") for hc, city in cities.items()}


# =============================================================================
# Refresh status (persists the last-refresh timestamp only)
# =============================================================================
def _read_snapshot() -> dict:
    """Read the refresh-status JSON, or {} if missing/corrupt."""
    if _SNAPSHOT_JSON.exists():
        try:
            return json.loads(_SNAPSHOT_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _write_snapshot(snap: dict) -> None:
    """Persist the refresh-status JSON (creates Data/ if needed)."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_JSON.write_text(json.dumps(snap, indent=2), encoding="utf-8")


# =============================================================================
# Main interface — pages call these
# =============================================================================
def get_scored_bookings(force_refresh: bool = False) -> pd.DataFrame:
    """Scored upcoming bookings in the canonical schema (real backend only).

    Dummy backend removed (v11). On any failure (no model on disk / no BigQuery
    creds) returns an EMPTY canonical frame so the app renders a graceful
    empty-state instead of crashing.
    """
    from . import real
    try:
        return real.get_scored_bookings(
            model_name=get_model(),
            horizon_days=GENERATION_HORIZON_DAYS,
            force_refresh=force_refresh,
        )
    except Exception as e:  # noqa: BLE001 — graceful empty-state
        print(f"get_scored_bookings: real backend unavailable ({type(e).__name__}: {e})")
        return pd.DataFrame(columns=list(S.COLUMNS))


def get_metadata() -> dict:
    """Data-status summary for the Home + Datenaktualisierung pages."""
    df = get_scored_bookings()
    today = pd.Timestamp.today().normalize()
    conf = derive.confirmed(df)
    # Already-canceled rows (excluded from most views).
    canceled = df[df[S.STATUS] == S.STATUS_CANCELED] if S.STATUS in df else df.iloc[0:0]
    # Future arrivals only.
    upcoming = conf[conf[S.ARRIVAL_DATE] >= today]
    snap = _read_snapshot()
    return {
        "mode": mode(),
        "model": get_model() or "auto (best AP + Brier gate)",
        "refreshed_at": snap.get("refreshed_at", datetime.now().astimezone().isoformat()),
        "reservations": {"rows": int(len(df))},
        "confirmed": {"rows": int(len(conf))},
        "canceled": {"rows": int(len(canceled))},
        "upcoming": {"rows": int(len(upcoming))},
        "properties": sorted(df[S.HOTEL_CODE].dropna().unique().tolist()),
        "window": {
            "earliest": str(conf[S.ARRIVAL_DATE].min().date()) if len(conf) else None,
            "latest": str(conf[S.ARRIVAL_DATE].max().date()) if len(conf) else None,
        },
        "high_risk": int((upcoming[S.CANCEL_PROBA] >= S.HIGH_THR).sum()) if len(upcoming) else 0,
        "cancel_rate": round(float(upcoming[S.CANCEL_PROBA].mean()), 3) if len(upcoming) else 0.0,
    }


def refresh(**kwargs) -> dict:
    """Re-pull upcoming arrivals from BigQuery + re-score, then return metadata."""
    # Clear location/occupancy caches so a refresh re-reads the source of truth.
    units_by_hotel.cache_clear(); city_by_hotel.cache_clear(); hotel_labels.cache_clear()
    from . import occupancy
    _c = getattr(occupancy._fallback_perf, "cache_clear", None)
    if _c:
        _c()
    from . import real
    # Force a fresh BigQuery pull + scoring with the current model.
    try:
        real.get_scored_bookings(model_name=get_model(), force_refresh=True)
        from .real import _import_src
        _import_src().load_property_performance(force_refresh=True)
    except Exception as e:  # noqa: BLE001 — table/creds optional
        print(f"refresh: real re-pull skipped ({e})")
    _write_snapshot({
        "refreshed_at": datetime.now().astimezone().isoformat(),
        "horizon_days": GENERATION_HORIZON_DAYS,
    })
    return get_metadata()
