# dash_app/backend/locations.py
# ---------------------------------------------------------------------------
# Real location table from configs/locations.yaml. Replaces the dummy loader
# (`dummy._load_locations`) that was removed in v11. Returns the same
# (hotel_code, city, units_total) tuples the location helpers expect, so the
# backend keeps working with the dummy backend gone.
# ---------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache

import yaml

from .. import config as CFG

_LOCATIONS_YAML = CFG.REPO_ROOT / "configs" / "locations.yaml"


@lru_cache(maxsize=1)
def _load_locations() -> list[tuple[str, str, int]]:
    """(hotel_code, city, units_total) per property from configs/locations.yaml.

    Returns [] if the file is missing — callers then fall back to the real
    property-performance universe (BigQuery) or a graceful empty state.
    """
    if not _LOCATIONS_YAML.exists():
        return []
    data = yaml.safe_load(_LOCATIONS_YAML.read_text(encoding="utf-8")) or {}
    out: list[tuple[str, str, int]] = []
    for loc in data.get("locations", []):
        hc = loc.get("hotel_code")
        if not hc:
            continue
        out.append((str(hc), str(loc.get("city", "")), int(loc.get("units_total") or 0)))
    return out
