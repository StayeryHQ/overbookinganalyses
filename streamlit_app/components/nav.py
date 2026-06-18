"""Landing-Page-Bausteine: Navigations-Cards + Platzhalter-Panel.

Reine Render-Helfer auf Basis der CSS-Klassen aus ``brand.py``
(``stayery-navcard`` / ``stayery-placeholder``).
"""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


def nav_card(
    *,
    page: str,
    icon: str,
    kicker: str,
    title: str,
    desc: str,
    link_label: str,
    status: str = "ready",
    status_label: str | None = None,
) -> None:
    """Eine klickbare Navigations-Kachel rendern.

    Args:
        page: Pfad zur Ziel-Page relativ zum App-Root (z. B. "pages/1_...py").
        icon: Emoji/Symbol oben in der Kachel.
        kicker: kleiner Uppercase-Übertitel.
        title: Kachel-Titel.
        desc: Kurzbeschreibung.
        link_label: Text des page_link-Buttons darunter.
        status: "ready" (grün) oder "soon" (grau).
        status_label: optionaler Text fürs Status-Pill (Default je nach status).
    """
    if status_label is None:
        status_label = "Verfügbar" if status == "ready" else "In Arbeit"
    st.markdown(
        f'<div class="stayery-navcard">'
        f'<div class="nc-icon">{icon}</div>'
        f'<div class="nc-kicker">{kicker}</div>'
        f'<div class="nc-title">{title}</div>'
        f'<div class="nc-desc">{desc}</div>'
        f'<span class="nc-status {status}">{status_label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.page_link(page, label=link_label)


def placeholder_panel(
    *,
    title: str,
    intro: str,
    planned: Iterable[str] | None = None,
    badge: str = "Nächster Schritt",
) -> None:
    """Gestrichelte Platzhalter-Box für noch nicht befüllte Subpages."""
    items = ""
    if planned:
        lis = "".join(f"<li>{p}</li>" for p in planned)
        items = f"<ul>{lis}</ul>"
    st.markdown(
        f'<div class="stayery-placeholder">'
        f'<span class="ph-badge">{badge}</span>'
        f'<div class="ph-title">{title}</div>'
        f'<div class="ph-body">{intro}{items}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
