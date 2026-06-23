# dash_app/components/__init__.py
# ---------------------------------------------------------------------------
# Re-exports the reusable UI helpers (simplifies to one import)
# ---------------------------------------------------------------------------

from __future__ import annotations

# Raw SVG icon markup.
from .icons import icon_markup

# The persistent sidebar nav.
from .sidebar import sidebar

# Layout / content helpers (Dash component factories).
from .ui import (
    alert,
    caption,
    explain,
    hero,
    metric,
    metric_row,
    nav_card,
    nav_card_grid,
    placeholder_panel,
    section,
)

# Public names of this package.
__all__ = [
    "hero",
    "alert",
    "metric",
    "metric_row",
    "section",
    "explain",
    "caption",
    "nav_card",
    "nav_card_grid",
    "placeholder_panel",
    "sidebar",
    "icon_markup",
]
