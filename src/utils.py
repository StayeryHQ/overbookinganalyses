# ---------------------------------------------------------------------------
# Generic helpers: brand colours (plotly-friendly), location reference table,
# small plotting utilities.
# ---------------------------------------------------------------------------

"""Theming, palette helpers and the locations reference table."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import brand_config_path

# =============================================================================
# Brand config / palette
# =============================================================================


@lru_cache(maxsize=1)
def load_brand_config() -> dict[str, Any]:
    """Load the Stayery brand spec from YAML (cached per process)."""
    path: Path = brand_config_path()
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _color_lookup() -> dict[str, str]:
    """Flatten the {core, supporting} palettes into one name->hex dict."""
    cfg = load_brand_config()
    return {**cfg["colors"]["core"], **cfg["colors"]["supporting"]}


def color(name: str) -> str:
    """Return a single Stayery color hex by its name."""
    lookup = _color_lookup()
    if name not in lookup:
        raise KeyError(f"Unknown Stayery color '{name}'. Known: {sorted(lookup)}")
    return lookup[name]


def categorical_palette(n: int | None = None) -> list[str]:
    """Return the Stayery categorical palette as a list of hex strings."""
    cfg = load_brand_config()
    lookup = _color_lookup()
    palette = [lookup[name] for name in cfg["categorical_order"]]
    if n is None:
        return palette
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def diverging_triplet() -> tuple[str, str, str]:
    """Return (negative, neutral, positive) hex triplet."""
    cfg = load_brand_config()
    lookup = _color_lookup()
    div = cfg["diverging"]
    return lookup[div["negative"]], lookup[div["neutral"]], lookup[div["positive"]]



def benchmark_overbooking_allowance(units_total: int) -> int:
    """Business rule: how many overbookings does the benchmark permit?
    Under 50 rooms -> 2 allowed
    50 rooms or more -> 4 allowed
    """
    return 4 if units_total >= 50 else 2


# =============================================================================
# Room-type capacities (for the Occupancy dashboard's room-type sub-view)
# =============================================================================
from .paths import configs_dir  # noqa: E402  (kept local to this section)

ROOM_TYPE_CAPACITY_FILE = "room_type_capacity.yaml"


@lru_cache(maxsize=1)
def load_room_type_capacity() -> dict[str, dict[str, int]]:
    """Load configs/room_type_capacity.yaml -> {property_name: {unitGroup_name: capacity}}.

    This file is HAND-MAINTAINED (real per-room-type capacities aren't in either
    BigQuery table). Entries whose capacity is still null/blank are dropped, so the
    room-type view simply omits a capacity reference line for those types instead of
    drawing a wrong one. Returns {} if the file is missing.
    """
    path: Path = configs_dir() / ROOM_TYPE_CAPACITY_FILE
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    caps = raw.get("capacities", raw)  # allow either a top-level 'capacities:' or a flat map
    out: dict[str, dict[str, int]] = {}
    for prop, groups in (caps or {}).items():
        if not isinstance(groups, dict):
            continue
        clean = {str(g): int(v) for g, v in groups.items() if v is not None and str(v) != ""}
        if clean:
            out[str(prop)] = clean
    return out


RISK_BUCKETS_FILE = "risk_buckets.yaml"

# Fallback if the config file is missing — same PLACEHOLDER values, so the column is
# never silently empty. Replace via configs/risk_buckets.yaml.
_RISK_BUCKETS_DEFAULT: dict = {
    "low_max": 0.20, "high_min": 0.50,
    "labels": {"low": "Low", "medium": "Medium", "high": "High"},
}


@lru_cache(maxsize=1)
def load_risk_buckets() -> dict:
    """Load configs/risk_buckets.yaml -> {low_max, high_min, labels{low,medium,high}}.

    These are the (currently PLACEHOLDER) cancel-probability cut points for the
    booking table's Risk column. Falls back to sane defaults if the file is absent.
    """
    path: Path = configs_dir() / RISK_BUCKETS_FILE
    if not path.exists():
        return dict(_RISK_BUCKETS_DEFAULT)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    out = dict(_RISK_BUCKETS_DEFAULT)
    out.update({k: cfg[k] for k in ("low_max", "high_min", "labels") if k in cfg})
    return out


def risk_label(p: float | None, cfg: dict | None = None) -> str:
    """Map a cancel probability to its bucket LABEL using the config thresholds.

    Returns "" for a missing/NaN probability (so the column is blank only when there
    is genuinely no score, never because the mapping was forgotten).
    """
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p != p:  # NaN
        return ""
    cfg = cfg or load_risk_buckets()
    labels = cfg.get("labels", _RISK_BUCKETS_DEFAULT["labels"])
    if p < cfg.get("low_max", 0.20):
        return labels.get("low", "Low")
    if p < cfg.get("high_min", 0.50):
        return labels.get("medium", "Medium")
    return labels.get("high", "High")
