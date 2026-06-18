"""Feine Inline-SVG-Icons — bewusst statt Emojis.

Stroke-basiert, ``currentColor``, dezent. Werden in Nav-Cards und
Section-Köpfen verwendet. Jede Funktion gibt fertiges SVG-Markup zurück.
"""

from __future__ import annotations

_INK = "#111111"
_YELLOW = "#FFE650"

_PATHS: dict[str, str] = {
    # Belegung / Kalender
    "calendar": '<rect x="3" y="4.5" width="18" height="16" rx="2.5"/>'
                '<path d="M3 9h18M8 2.5v4M16 2.5v4"/>',
    # Vorhersage / Trend
    "trend": '<path d="M3 17l5.5-5.5 3.5 3.5L21 7"/><path d="M21 12V7h-5"/>',
    # Datenaktualisierung
    "refresh": '<path d="M20 11a8 8 0 1 0-.6 4"/><path d="M20 4v5h-5"/>',
    # Auslastung / Gebäude
    "building": '<rect x="4" y="3" width="16" height="18" rx="1.5"/>'
                '<path d="M8 7h2M14 7h2M8 11h2M14 11h2M8 15h2M14 15h2M10 21v-3h4v3"/>',
    # Tabelle
    "table": '<rect x="3" y="4.5" width="18" height="15" rx="2"/>'
             '<path d="M3 10h18M3 15h18M9 4.5v15M15 4.5v15"/>',
    # Filter
    "filter": '<path d="M3 5h18l-7 8v6l-4-2v-4z"/>',
    # Risiko / Blitz
    "bolt": '<path d="M13 2 4 14h6l-1 8 9-12h-6z"/>',
    # Info
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/>',
    # Pfeile
    "arrow_right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "arrow_left": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
}


def icon(name: str, size: int = 24, color: str = _INK, stroke: float = 1.6) -> str:
    """SVG-Icon-Markup zurückgeben."""
    body = _PATHS.get(name, _PATHS["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" '
        f'style="display:inline-block;vertical-align:middle;">{body}</svg>'
    )
