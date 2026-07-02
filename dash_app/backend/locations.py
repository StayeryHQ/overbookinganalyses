# dash_app/backend/locations.py
# Property (location) universe for the Dash app, backed by the SQL-derived
# property_performance_daily cache via src.property_universe(). Falls back to an
# empty list when the cache / BigQuery is unavailable, so the app shell still
# renders instead of crashing.

from __future__ import annotations

from functools import lru_cache

from src import property_universe


@lru_cache(maxsize=1)
def load_locations() -> list[dict[str, object]]:
    """Return one record per property: [{'propertyId': str, 'units': int}, ...].

    Source of truth is the local property_performance cache (build/refresh it with
    `python main.py refresh`). `units` = the property's most recent houseCount.
    Returns [] if the cache / BigQuery is unavailable — callers must handle empty.

    NOTE: this exposes `propertyId` (the performance table's property code), not the
    `property_name` used in the reservations table. The propertyId -> property_name
    mapping is still open (see audit_findings.md §3b/§6) and will be resolved when the
    property table is fully wired into the dashboard.

    Cached per process; call `load_locations.cache_clear()` after a data refresh.
    """
    df = property_universe()
    if df.empty:
        return []
    return df.to_dict("records")
