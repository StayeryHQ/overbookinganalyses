# dash_app/components/icons.py
# ---------------------------------------------------------------------------
# Fine inline-SVG icons, remdered via the `dangerously_allow_html` flag on a
# dash.html element, or wrap it with DangerouslySetInnerHTML. expose raw and wrapper
# ---------------------------------------------------------------------------

from __future__ import annotations

# Default stroke colour (brand ink).
_INK = "#111111"

# Stroke-based path data per icon name (24x24 viewBox, currentColor stroke).
_PATHS: dict[str, str] = {
    # Belegung / calendar
    "calendar": '<rect x="3" y="4.5" width="18" height="16" rx="2.5"/>'
                '<path d="M3 9h18M8 2.5v4M16 2.5v4"/>',
    # Vorhersage / trend
    "trend": '<path d="M3 17l5.5-5.5 3.5 3.5L21 7"/><path d="M21 12V7h-5"/>',
    # Datenaktualisierung / refresh
    "refresh": '<path d="M20 11a8 8 0 1 0-.6 4"/><path d="M20 4v5h-5"/>',
    # Auslastung / building
    "building": '<rect x="4" y="3" width="16" height="18" rx="1.5"/>'
                '<path d="M8 7h2M14 7h2M8 11h2M14 11h2M8 15h2M14 15h2M10 21v-3h4v3"/>',
    # Tabelle / table
    "table": '<rect x="3" y="4.5" width="18" height="15" rx="2"/>'
             '<path d="M3 10h18M3 15h18M9 4.5v15M15 4.5v15"/>',
    # Risiko / bolt
    "bolt": '<path d="M13 2 4 14h6l-1 8 9-12h-6z"/>',
    # Info
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/>',
}


def icon_markup(name: str, size: int = 24, color: str = _INK, stroke: float = 1.6) -> str:
    """Return raw SVG markup for the named icon (string)."""
    # Fall back to the info icon if the name is unknown.
    body = _PATHS.get(name, _PATHS["info"]) # info is the fallback!
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" '
        f'style="display:inline-block;vertical-align:middle;">{body}</svg>'
    )
