# ---------------------------------------------------------------------------
# Generic helpers: brand theming for matplotlib, location reference table,
# small plotting utilities.
# ---------------------------------------------------------------------------

"""Theming, palette helpers and the locations reference table."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib as mpl

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


def apply_stayery_style() -> None:
    """Apply the Stayery matplotlib style globally for the current session.

    Defensive against missing brand fonts (Neue Haas Grotesk is a paid
    Linotype font most laptops don't have). We silence the
    'findfont: Font family not found' warning so users don't get spammed,
    and we put always-available fallbacks at the END of the chain.
    """
    import logging

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
