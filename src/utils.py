# ---------------------------------------------------------------------------
# Generic helpers: brand colours (plotly-friendly), location reference table,
# small plotting utilities.
# ---------------------------------------------------------------------------

"""Theming, palette helpers and the locations reference table."""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paths import brand_config_path


# =============================================================================
# Timestamps — stored in UTC everywhere, DISPLAYED in local time everywhere.
# =============================================================================
def local_timezone():
    """The machine's local timezone (the app runs on the user's machine)."""
    return datetime.now().astimezone().tzinfo


def fmt_ts_local(value, fmt: str = "%b %d, %Y, %H:%M %Z") -> str | None:
    """Human-readable timestamp in LOCAL time, e.g. 'Jul 13, 2026, 14:30 CEST'.

    THE one display formatter for the app — never render raw UTC to users.
    Accepts ISO strings / datetimes / pd.Timestamps; naive input is treated as
    UTC (that is how everything is stored). Returns None if unparseable, so
    callers can show 'unavailable' instead of a wrong time.
    """
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert(local_timezone()).strftime(fmt)

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


def apply_stayery_style() -> None:
    """Apply the Stayery matplotlib style globally for the current session.

    For the matplotlib/seaborn notebooks (the Dash app itself uses Plotly via
    theme.brand_figure and does NOT need this). `matplotlib` is imported LAZILY inside the
    function, so importing `src`/`src.utils` never requires matplotlib — only calling this
    helper does. Install matplotlib in the notebook env if it isn't already (it ships as a
    dependency again; `uv sync`).

    Defensive against missing brand fonts (Neue Haas Grotesk is a paid Linotype font most
    laptops don't have). We silence the 'findfont: Font family not found' warning so users
    don't get spammed, and we put always-available fallbacks at the END of the chain.
    """
    import logging

    import matplotlib as mpl

    # Matplotlib emits this warning per missing font, per draw call.
    # Silence it once; the fallback chain still does its job.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    cfg = load_brand_config()
    lookup = _color_lookup()

    primary_chain = [cfg["typography"]["primary"]] + cfg["typography"][
        "primary_fallback"
    ]
    if "DejaVu Sans" not in primary_chain:
        primary_chain.append("DejaVu Sans")
    palette = categorical_palette()

    mpl.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": primary_chain,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "regular",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            # Color cycle
            "axes.prop_cycle": mpl.cycler(color=palette),
            # Backgrounds
            "figure.facecolor": lookup["white"],
            "axes.facecolor": lookup["white"],
            "savefig.facecolor": lookup["white"],
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": lookup["black"],
            "axes.linewidth": 1.0,
            # Grid
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            # Ticks
            "xtick.color": lookup["black"],
            "ytick.color": lookup["black"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Lines & markers
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            # Figure size / DPI
            "figure.figsize": (10, 5.5),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


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


def _normalize_room_type_label(value: str | None) -> str:
    """Canonicalise room-type labels so matching ignores case, spacing and separators."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(with|mit)\b", " ", text)
    text = text.replace("balkon", "balcony")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_room_type_capacity(
    capacities: dict[str, dict[str, int]],
    property_name: str | None,
    room_type: str | None,
) -> int | None:
    """Resolve a room-type capacity by exact match or a normalized alias match."""
    if not capacities or property_name is None or room_type is None:
        return None

    target_prop = _normalize_room_type_label(property_name)
    target_room = _normalize_room_type_label(room_type)
    if not target_prop or not target_room:
        return None

    prop_candidates = [
        prop for prop in capacities if _normalize_room_type_label(prop) == target_prop
    ]
    if not prop_candidates:
        return None

    groups = capacities[prop_candidates[0]]
    if str(room_type) in groups:
        return groups[str(room_type)]

    for label, value in groups.items():
        if _normalize_room_type_label(label) == target_room:
            return value
    return None


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
    caps = raw.get(
        "capacities", raw
    )  # allow either a top-level 'capacities:' or a flat map
    out: dict[str, dict[str, int]] = {}
    for prop, groups in (caps or {}).items():
        if not isinstance(groups, dict):
            continue
        clean = {
            str(g): int(v) for g, v in groups.items() if v is not None and str(v) != ""
        }
        if clean:
            out[str(prop)] = clean
    return out


RISK_BUCKETS_FILE = "risk_buckets.yaml"

# Fallback if the config file is missing — same PLACEHOLDER values, so the column is
# never silently empty. Replace via configs/risk_buckets.yaml.
_RISK_BUCKETS_DEFAULT: dict = {
    "low_max": 0.20,
    "high_min": 0.50,
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
