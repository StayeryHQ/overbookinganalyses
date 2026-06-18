"""Section-Wrapper & dynamisches Inhaltsverzeichnis.

Gespiegelt vom RevenueBlindSpots-Design:

  - ``section(N, title)``      — Context Manager, rendert immer (mit Anchor).
  - ``lazy_section(N, title)`` — Click-to-load (verhindert, dass schwere Charts
                                  auf jedem Rerun gerendert werden).
  - ``render_toc(entries)``    — klickbares TOC oben auf der Page.
  - Jede Sektion endet mit einem dezenten „↑ nach oben"-Link.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import streamlit as st

# ============================== Anchor IDs ================================
_TOP_ID = "page-top"


def _anchor_id(num: int | str) -> str:
    """Stabile, HTML-sichere Anchor-ID für eine Sektion."""
    s = str(num).replace(".", "-")
    return f"sec-{s}"


def _anchor_html(num: int | str) -> str:
    """Unsichtbarer <a id>-Anchor, auf den Scroll-Sprünge landen."""
    return f'<div id="{_anchor_id(num)}"></div>'


def _back_to_top_html() -> str:
    return f'<div class="stayery-totop"><a href="#{_TOP_ID}">↑ nach oben</a></div>'


# ============================== TOC =======================================
def render_toc(entries: Iterable[tuple[int | str, str]]) -> None:
    items = list(entries)
    if not items:
        return
    st.markdown(f'<div id="{_TOP_ID}"></div>', unsafe_allow_html=True)
    links = "".join(
        f'<a href="#{_anchor_id(n)}"><span class="num">{n}</span>{t}</a>' for n, t in items
    )
    st.markdown(
        f'<nav class="stayery-toc" aria-label="Inhalt">'
        f'<span class="stayery-toc-label">Inhalt</span>'
        f'<div class="stayery-toc-links">{links}</div>'
        f"</nav>",
        unsafe_allow_html=True,
    )


# ============================== Section wrappers ==========================
@contextmanager
def section(
    num: int | str | None,
    title: str,
    *,
    subtitle: str | None = None,
    description: str | None = None,
):
    if num is not None and num != 0:
        st.markdown(_anchor_html(num), unsafe_allow_html=True)
        header = f"{num} · {title}"
    else:
        header = title
    st.markdown(f"## {header}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    if description:
        st.markdown(description)
    yield
    if num is not None and num != 0:
        st.markdown(_back_to_top_html(), unsafe_allow_html=True)
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)


def lazy_section(
    num: int | str,
    title: str,
    *,
    subtitle: str | None = None,
    description: str | None = None,
) -> bool:
    key = _key(num)
    loaded = st.session_state.get(key, False)

    if loaded:
        st.markdown(_anchor_html(num), unsafe_allow_html=True)
        st.markdown(f"## {num} · {title}")
        if subtitle:
            st.markdown(f"*{subtitle}*")
        if description:
            st.markdown(description)
        return True

    st.markdown(_anchor_html(num), unsafe_allow_html=True)
    cols = st.columns([5, 1])
    cols[0].markdown(f"### {num} · {title}")
    if subtitle:
        cols[0].caption(subtitle)
    if cols[1].button("Laden", key=f"_btn_{key}", use_container_width=True):
        st.session_state[key] = True
        st.rerun()
    st.markdown(_back_to_top_html(), unsafe_allow_html=True)
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    return False


def section_loaded(num: int | str) -> bool:
    return bool(st.session_state.get(_key(num), False))


def preload_all_button(
    section_nums: list[int | str], label: str = "Alle Sektionen laden"
) -> None:
    """Setzt das 'loaded'-Flag für mehrere Sektionen auf einmal."""
    if st.button(
        label, use_container_width=True, help="Lädt alle verbleibenden Sektionen auf einmal."
    ):
        for n in section_nums:
            st.session_state[_key(n)] = True
        st.rerun()


def _key(num) -> str:
    return f"_loaded_sec_{num}"
