"""Alarm- / Highlight-Boxen für die Pages — emoji-frei.

Vier Stile, jeweils farbiger linker Rand + kleiner Text-Tag (kein Emoji):
  - alert    (rot)   kritische Hinweise
  - warning  (gelb)  auffällige Beobachtungen
  - info     (blau)  neutrale Hinweise
  - success  (grün)  positive Findings
"""

from __future__ import annotations

import streamlit as st

_STYLES = {
    "alert": ("#fde7e3", "#9a2316", "ACHTUNG"),
    "warning": ("#fff5d6", "#9a6f00", "HINWEIS"),
    "info": ("#e3eaf5", "#1f3d7a", "INFO"),
    "success": ("#e3f5ea", "#137a3a", "OK"),
}


def alert_card(message: str, kind: str = "info", *, title: str | None = None) -> None:
    """Farbige Highlight-Box rendern.

    Args:
        message: Textkörper (Plain-String oder Markdown).
        kind: 'alert', 'warning', 'info' oder 'success'.
        title: optionaler fetter Titel über dem Body.
    """
    bg, fg, tag = _STYLES.get(kind, _STYLES["info"])
    head = title or tag.capitalize()
    body = message.replace("\n", "<br>")
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {fg};'
        f"color:#1a1a1a;padding:10px 14px;border-radius:8px;margin:6px 0;"
        f'font-size:0.92em;line-height:1.45;">'
        f'<span style="display:inline-block;font-size:0.66em;font-weight:700;'
        f"letter-spacing:0.10em;text-transform:uppercase;color:{fg};"
        f'margin-bottom:3px;">{head}</span><br>{body}</div>',
        unsafe_allow_html=True,
    )


def alert_cards(highlights: list[dict]) -> None:
    """Vertikaler Stack von Alerts aus einer Liste von Dicts."""
    if not highlights:
        alert_card("Keine Auffälligkeiten gefunden", kind="success")
        return
    for h in highlights:
        alert_card(h.get("message", ""), kind=h.get("kind", "info"), title=h.get("title"))
