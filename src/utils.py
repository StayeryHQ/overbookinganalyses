# ---------------------------------------------------------------------------
# Generic helpers: brand colours (plotly-friendly), location reference table,
# small plotting utilities.
# ---------------------------------------------------------------------------

"""Theming, palette helpers and the locations reference table."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paths import brand_config_path


# =============================================================================
# Timestamps  stored in UTC everywhere, DISPLAYED in local time everywhere.
# =============================================================================
def local_timezone():
    """The machine's local timezone (the app runs on the user's machine)."""
    return datetime.now().astimezone().tzinfo


def fmt_ts_local(value, fmt: str = "%b %d, %Y, %H:%M %Z") -> str | None:
    """Human-readable timestamp in LOCAL time, e.g. 'Jul 13, 2026, 14:30 CEST'.

    THE one display formatter for the app  never render raw UTC to users.
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
    function, so importing `src`/`src.utils` never requires matplotlib  only calling this
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
# Risk buckets (booking-table Risk column)  ONE cost-based rule
# =============================================================================
# Low / Medium / High are derived from a single COST-BASED decision threshold
# plus a FIXED high-risk cutoff:
#     p <  threshold          -> Low
#     threshold <= p < 0.85   -> Medium
#     p >= 0.85               -> High   (always, independent of the threshold)
# `threshold` is the cost-optimal decision threshold (src.scoring); it moves with
# the walk/empty costs entered in the app. The 0.85 High cutoff is fixed here so
# there is exactly one place to change it.
from .paths import configs_dir  # noqa: E402  (kept local to this section)

RISK_BUCKETS_FILE = "risk_buckets.yaml"

# Fixed High-risk cutoff  a booking at/above this cancel probability is ALWAYS
# High, regardless of the cost-based threshold. Single source of truth.
HIGH_RISK_CUTOFF: float = 0.85

# Fallback if the config file is missing  so labels are never silently empty.
_RISK_BUCKETS_DEFAULT: dict = {
    "high_cutoff": HIGH_RISK_CUTOFF,
    "labels": {"low": "Low", "medium": "Medium", "high": "High"},
}


@lru_cache(maxsize=1)
def load_risk_buckets() -> dict:
    """Load configs/risk_buckets.yaml -> {high_cutoff, labels{low,medium,high}}.

    Only the fixed High cutoff and the display labels live in config now; the
    Low/Medium boundary is the dynamic cost-based threshold (not a config value).
    Falls back to sane defaults if the file is absent.
    """
    path: Path = configs_dir() / RISK_BUCKETS_FILE
    if not path.exists():
        return dict(_RISK_BUCKETS_DEFAULT)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    out = dict(_RISK_BUCKETS_DEFAULT)
    for k in ("high_cutoff", "labels"):
        if k in cfg:
            out[k] = cfg[k]
    return out


def risk_label_cost(p: float | None, threshold: float,
                    high_cut: float = HIGH_RISK_CUTOFF,
                    labels: dict | None = None) -> str:
    """Map a cancel probability to Low / Medium / High using the cost-based rule.

    p >= high_cut -> High (always); threshold <= p < high_cut -> Medium; else Low.
    Returns "" for a missing/NaN probability (blank only when there is genuinely no
    score). If threshold >= high_cut the Medium band vanishes (High or Low only).
    """
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p != p:  # NaN
        return ""
    labels = labels or load_risk_buckets().get("labels", _RISK_BUCKETS_DEFAULT["labels"])
    if p >= float(high_cut):
        return labels.get("high", "High")
    if p >= float(threshold):
        return labels.get("medium", "Medium")
    return labels.get("low", "Low")
