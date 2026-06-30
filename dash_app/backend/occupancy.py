# dash_app/backend/occupancy.py
# ---------------------------------------------------------------------------
# Occupancy + daily operations source = the BigQuery table
# `reporting.property_performance_daily` (occupancy, houseCount, soldCount,
# outOfOrderCount, departuresCount, noShowsCount, cancellationsCount), keyed by
# propertyId (== the apaleo code == reservations.property_id == hotel_code).
#
# This module is the dash_app adapter:
#   * REAL  -> src.load_property_performance() (cached parquet / BigQuery).
#   * FALLBACK -> a synthetic, seeded occupancy frame so the page renders with no
#              BigQuery access.
# Revenue columns are never pulled (the src loader selects an allow-list).
#
# IMPORTANT (flagged): the occupancy/ops figures are DAILY ACTUALS, so future
# business days may be absent. derive.daily_grid uses these where a (property,
# date) row exists and falls back to a booking-derived occupancy proxy otherwise
# — so the 14-day forward raster stays populated either way.
# ---------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import schema as S
from .. import config as CFG

# The canonical (non-revenue) columns we work with downstream.
PERF_COLS = ["propertyId", "businessDay", "houseCount", "soldCount", "outOfOrderCount",
             "departuresCount", "noShowsCount", "cancellationsCount", "occupancyPercentage"]


def get_perf(force_refresh: bool = False) -> pd.DataFrame:
    """Property-performance (occupancy) frame with PERF_COLS.

    Real apaleo/BigQuery table when available; otherwise a synthetic occupancy
    fallback so the dashboard's occupancy gating still renders (this is an
    OCCUPANCY proxy only — never the cancellation model, which is always real).
    """
    try:
        from .real import _import_src
        src = _import_src()
        df = src.load_property_performance(force_refresh=force_refresh)
        # Keep only the columns we use, in canonical order (robust to drift).
        return df[[c for c in PERF_COLS if c in df.columns]].copy()
    except Exception as e:  # noqa: BLE001 — no creds/table/offline -> synth fallback
        print(f"occupancy: real performance table unavailable ({e}); using fallback.")
    return _fallback_perf(force_refresh=force_refresh)


@lru_cache(maxsize=4)
def _fallback_perf(force_refresh: bool = False) -> pd.DataFrame:
    """Synthetic OCCUPANCY frame (locations × business days) matching the real
    schema, so occupancy gating renders when the apaleo table is unavailable.
    `force_refresh` is part of the cache key so a refresh yields a fresh frame.
    NB: occupancy proxy only — the cancellation model/probabilities are real.
    """
    from .locations import _load_locations
    today = pd.Timestamp.today().normalize()
    # Business days from 3 days back to 20 ahead (so 'past actuals' + some forward).
    days = [today + pd.Timedelta(days=i) for i in range(-3, 21)]
    rng = np.random.RandomState(7 if not force_refresh else int(pd.Timestamp.now().value % 9973))
    rows = []
    for hc, _city, units in _load_locations():             # each property
        units = int(units) or 60
        for d in days:                                     # each business day
            occ = float(np.clip(rng.uniform(0.55, 0.97), 0, 1.2))   # occupancy fraction
            sold = int(round(occ * units))                 # sold units
            ooo = int(rng.binomial(units, 0.02))           # out-of-order units
            rows.append({
                "propertyId": hc, "businessDay": d, "houseCount": units,
                "soldCount": sold, "outOfOrderCount": ooo,
                "departuresCount": int(rng.poisson(max(1, sold * 0.22))),
                "noShowsCount": int(rng.poisson(0.4)),
                "cancellationsCount": int(rng.poisson(1.2)),
                "occupancyPercentage": round(occ * 100, 1),  # 0..100 like apaleo
            })
    return pd.DataFrame(rows)


def units_from_universe() -> dict[str, int]:
    """{propertyId: units} from the REAL performance table, else {} (caller falls back).

    This is what replaces configs/locations.yaml: new propertyIds appear here
    automatically. Empty when the table is unavailable (no creds / offline).
    """
    try:
        from .real import _import_src
        uni = _import_src().property_universe()
        if uni is None or uni.empty:
            return {}
        return {str(r.propertyId): int(r.units) for r in uni.itertuples()}
    except Exception:  # noqa: BLE001
        return {}
