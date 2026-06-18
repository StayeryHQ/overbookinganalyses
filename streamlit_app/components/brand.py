"""Stayery brand → Streamlit-UI (CSS-Injection + Hero + Nav-Cards).

Gespiegelt vom Design der RevenueBlindSpots-App, damit Overbooking- und
Revenue-App konsistent aussehen. Bewusst **self-contained**: liest die
Brand-Spec direkt aus ``configs/stayery_brand.yaml`` und hängt NICHT am
``src``-Package (das matplotlib/xgboost/joblib mitzieht). So bleibt die UI
leichtgewichtig und unabhängig von den ML-Notebooks.

Brand-System (Single Source of Truth):
  • ``configs/stayery_brand.yaml``        = Brand-Spec (Farben, Fonts).
  • ``streamlit_app/components/brand.py`` = liest YAML → generiert CSS.

Brand-Farbwert ändern → nur in der YAML.

Fonts: **Neue Haas Grotesk Display Pro** + **Topol** laufen über ``@font-face``
aus ``streamlit_app/static/fonts/``. Ohne die Files fällt der Stack auf
System-Fonts (Helvetica Neue → Helvetica → Arial). Keine externen Requests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

import streamlit as st
import yaml

# Repo-Root = OverbookingAnalyse/ (zwei Ebenen über dieser Datei:
# components/ → streamlit_app/ → repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRAND_CONFIG = _REPO_ROOT / "configs" / "stayery_brand.yaml"


@lru_cache(maxsize=1)
def load_brand_config() -> dict[str, Any]:
    """Brand-Spec aus der YAML laden (pro Prozess gecached)."""
    with _BRAND_CONFIG.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# =============================================================================
# UI-Neutrals (warm-graue Töne, die das Gelb tragen)
# =============================================================================
_UI = {
    "ink": "#111111",        # primary text
    "ink_soft": "#444444",
    "caption": "#888888",
    "muted": "#b8b6a8",
    "bg": "#FAFAF5",         # soft off-white (sidebar, hover bg)
    "bg_warm": "#FFFCF0",    # textarea (very subtle warm)
    "border": "#ECEAE0",     # subtle dividers / card borders
    "border_soft": "#F2F0E6",
}


def _brand_tokens() -> dict[str, str]:
    """Brand-Farben aus der YAML mit den UI-Neutrals mischen."""
    cfg = load_brand_config()
    core = cfg["colors"]["core"]
    return {
        "yellow": core["yellow"],
        "black": core["black"],
        "white": core["white"],
        **_UI,
    }


# =============================================================================
# CSS-Template — Token-Substitution via string.Template
# =============================================================================
_BRAND_CSS = Template(r"""
<style>
/* -------------------------------------------------------------------------
   0 — Brand-Fonts aus streamlit_app/static/fonts/. Fehlen die OTF-Files,
       greift der System-Font-Fallback. Keine externen Requests.
   ------------------------------------------------------------------------- */
@font-face {
    font-family: "Neue Haas Grotesk Display Pro";
    src: url("./app/static/fonts/NeueHaasGroteskDisplay-Regular.otf") format("opentype");
    font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
    font-family: "Neue Haas Grotesk Display Pro";
    src: url("./app/static/fonts/NeueHaasGroteskDisplay-Italic.otf") format("opentype");
    font-weight: 400; font-style: italic; font-display: swap;
}
@font-face {
    font-family: "Neue Haas Grotesk Display Pro";
    src: url("./app/static/fonts/NeueHaasGroteskDisplay-Medium.otf") format("opentype");
    font-weight: 500; font-style: normal; font-display: swap;
}
@font-face {
    font-family: "Neue Haas Grotesk Display Pro";
    src: url("./app/static/fonts/NeueHaasGroteskDisplay-MediumItalic.otf") format("opentype");
    font-weight: 500; font-style: italic; font-display: swap;
}
@font-face {
    font-family: "Neue Haas Grotesk Display Pro";
    src: url("./app/static/fonts/NeueHaasGroteskDisplay-Bold.otf") format("opentype");
    font-weight: 700; font-style: normal; font-display: swap;
}
@font-face {
    font-family: "Neue Haas Grotesk Display Pro";
    src: url("./app/static/fonts/NeueHaasGroteskDisplay-BoldItalic.otf") format("opentype");
    font-weight: 700; font-style: italic; font-display: swap;
}
@font-face {
    font-family: "Topol";
    src: url("./app/static/fonts/Topol-Bold.otf") format("opentype");
    font-weight: 700; font-style: normal; font-display: swap;
}

/* -------------------------------------------------------------------------
   1 — Grundlayout
   ------------------------------------------------------------------------- */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

html, body, [class*="css"], .stMarkdown, .stText, .stMetric, button, input, select, textarea {
    font-family: "Neue Haas Grotesk Display Pro",
                 "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* -------------------------------------------------------------------------
   2 — Headings — bold, mit dezenter Yellow-Akzent-Linie unten
   ------------------------------------------------------------------------- */
h1 {
    color: ${ink};
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 2.4rem !important;
    margin-bottom: 0.6rem !important;
}
h2 {
    color: ${ink};
    font-weight: 600;
    letter-spacing: -0.01em;
    font-size: 1.55rem !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid ${border};
    position: relative;
}
h2::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -1px;
    width: 48px;
    height: 3px;
    background: ${yellow};
    border-radius: 1px;
}
h3 { color: #2a2a2a; font-weight: 600; margin-top: 1.0rem; margin-bottom: 0.4rem; }
h4 { color: #2a2a2a; font-weight: 600; margin-top: 0.6rem; margin-bottom: 0.3rem; }

/* -------------------------------------------------------------------------
   3 — Buttons
   ------------------------------------------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button,
[data-testid^="baseButton-"] {
    background: ${white} !important;
    color: ${ink} !important;
    border: 1px solid ${ink} !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    transition: transform .22s cubic-bezier(.22,1,.36,1),
                background .22s cubic-bezier(.22,1,.36,1),
                color .18s ease,
                border-color .18s ease,
                box-shadow .25s cubic-bezier(.22,1,.36,1);
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid^="baseButton-"]:hover {
    background: #FFFBE3 !important;
    color: ${ink} !important;
    border-color: ${ink} !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(255,230,80,0.40), 0 2px 4px rgba(0,0,0,0.08) !important;
}
.stButton > button:active,
.stDownloadButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
    transition-duration: .06s;
}
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,230,80,0.55), 0 4px 12px rgba(0,0,0,0.08) !important;
}
/* Primary buttons (type="primary") = Yellow CTA */
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primaryFormSubmit"],
[data-testid="baseButton-primary"],
[data-testid="baseButton-primaryFormSubmit"] {
    background: ${yellow} !important;
    color: ${ink} !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="baseButton-primaryFormSubmit"]:hover {
    background: #FFF1A8 !important;
    color: ${ink} !important;
    border-color: ${ink} !important;
    box-shadow: 0 8px 18px rgba(255,230,80,0.55), 0 2px 5px rgba(0,0,0,0.10) !important;
}
[data-testid="stSidebar"] .stFormSubmitButton > button {
    padding: 0.45rem 0.9rem !important;
    font-size: 0.82rem;
}

/* -------------------------------------------------------------------------
   4 — Sidebar
   ------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background: ${bg};
    border-right: 1px solid ${border};
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    border-bottom: none;
    padding-bottom: 0;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: ${caption} !important;
    font-weight: 600;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2::after { display: none; }
[data-testid="stSidebar"] a[aria-current="page"] {
    background: ${yellow} !important;
    color: ${ink} !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] hr { margin: 1.5rem 0 !important; }

/* -------------------------------------------------------------------------
   5 — Metric Cards
   ------------------------------------------------------------------------- */
[data-testid="stMetric"] {
    background: ${white};
    border: 1px solid ${border};
    padding: 1rem 1.2rem;
    border-radius: 14px;
    transition: border-color .22s ease,
                transform .22s cubic-bezier(.22,1,.36,1),
                box-shadow .25s cubic-bezier(.22,1,.36,1);
    box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}
[data-testid="stMetric"]:hover {
    border-color: ${ink};
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    color: #6a6a6a;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem !important;
}
[data-testid="stMetricValue"] {
    color: ${ink} !important;
    font-weight: 700 !important;
    font-size: 1.85rem !important;
    letter-spacing: -0.01em;
    line-height: 1.15;
}
[data-testid="stMetricDelta"] {
    font-weight: 500;
    font-size: 0.82rem !important;
}

/* -------------------------------------------------------------------------
   6 — Tables
   ------------------------------------------------------------------------- */
[data-testid="stDataFrame"] {
    border: 1px solid ${border};
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead tr th {
    background: ${bg};
    color: ${ink};
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* -------------------------------------------------------------------------
   7 — Inputs / Selects
   ------------------------------------------------------------------------- */
.stTextArea textarea {
    border: 1px solid ${border} !important;
    background: ${bg_warm};
    border-radius: 12px;
    font-family: "Neue Haas Grotesk Display Pro", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    color: #222;
    transition: border-color .15s ease, box-shadow .2s ease;
}
.stTextArea textarea:focus {
    border-color: ${ink} !important;
    box-shadow: 0 0 0 1px ${ink} !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-baseweb="select"] > div {
    border-radius: 10px !important;
}

/* -------------------------------------------------------------------------
   8 — Section dividers
   ------------------------------------------------------------------------- */
hr {
    border: none;
    border-top: 1px solid ${border};
    margin: 1.2rem 0 !important;
}
[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }
[data-testid="stImage"] { margin: 0.3rem 0 !important; }
[data-testid="stMarkdownContainer"] p { margin-bottom: 0.4rem; }
.element-container { margin-bottom: 0.4rem !important; }

/* -------------------------------------------------------------------------
   9 — Status-Container
   ------------------------------------------------------------------------- */
[data-testid="stStatusWidget"] {
    background: ${bg};
    border-left: 3px solid ${yellow};
    border-radius: 10px;
}

/* -------------------------------------------------------------------------
   10 — TOC
   ------------------------------------------------------------------------- */
.stayery-toc {
    margin: 0 0 2rem 0;
    padding: 1rem 0 1.1rem 0;
    border-top: 1px solid ${border};
    border-bottom: 1px solid ${border};
    display: flex;
    flex-wrap: wrap;
    gap: 1.3rem 0;
    align-items: baseline;
}
.stayery-toc-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: ${caption};
    font-weight: 600;
    margin-right: 1.5rem;
    padding-top: 0.05rem;
}
.stayery-toc-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.6rem;
    flex: 1;
}
.stayery-toc a {
    text-decoration: none;
    color: ${ink};
    font-size: 0.88rem;
    font-weight: 500;
    border-bottom: 1px solid transparent;
    padding-bottom: 1px;
    transition: color .12s ease, border-color .12s ease;
    cursor: pointer;
}
.stayery-toc a:hover { border-bottom-color: ${yellow}; }
.stayery-toc .num {
    color: ${muted};
    font-variant-numeric: tabular-nums;
    margin-right: 0.35rem;
    font-weight: 400;
}

/* -------------------------------------------------------------------------
   11 — Back-to-top link
   ------------------------------------------------------------------------- */
.stayery-totop {
    text-align: right;
    margin-top: 0.6rem;
    font-size: 0.75rem;
}
.stayery-totop a {
    color: #b0b0b0;
    text-decoration: none;
    transition: color .12s ease;
    cursor: pointer;
}
.stayery-totop a:hover { color: ${ink}; }

/* -------------------------------------------------------------------------
   12 — Expander
   ------------------------------------------------------------------------- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    font-weight: 500;
    color: ${ink_soft};
    background: ${bg};
    border: 1px solid ${border};
    border-radius: 10px;
}
.streamlit-expanderHeader:hover, [data-testid="stExpander"] summary:hover {
    color: ${ink};
    background: ${border_soft};
}

/* -------------------------------------------------------------------------
   13 — Hero (Editorial-Kopf jeder Page)
   ------------------------------------------------------------------------- */
.stayery-hero {
    padding: 1.6rem 0 0.6rem;
    border-bottom: 1px solid ${border};
    margin-bottom: 1.6rem;
}
.stayery-hero-eyebrow {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: ${caption};
    font-weight: 600;
    margin-bottom: 0.3rem;
    display: inline-block;
    position: relative;
    padding-left: 18px;
}
.stayery-hero-eyebrow::before {
    content: "";
    position: absolute;
    left: 0; top: 50%;
    width: 12px; height: 2px;
    background: ${yellow};
    transform: translateY(-50%);
}
.stayery-hero-subtitle {
    color: ${ink_soft};
    font-size: 1.05rem;
    margin-top: 0.2rem;
    max-width: 720px;
}

/* -------------------------------------------------------------------------
   14 — Nav-Cards (Landing-Page Kacheln zu den Bereichen)
   ------------------------------------------------------------------------- */
.stayery-navcard {
    border: 1px solid ${border};
    border-radius: 16px;
    padding: 1.25rem 1.3rem 1.1rem;
    background: ${white};
    height: 100%;
    transition: transform .22s cubic-bezier(.22,1,.36,1),
                box-shadow .25s cubic-bezier(.22,1,.36,1),
                border-color .2s ease;
    position: relative;
    overflow: hidden;
}
.stayery-navcard::before {
    content: "";
    position: absolute;
    left: 0; top: 0;
    width: 4px; height: 100%;
    background: ${yellow};
    opacity: 0;
    transition: opacity .2s ease;
}
.stayery-navcard:hover {
    transform: translateY(-4px);
    border-color: ${ink};
    box-shadow: 0 12px 28px rgba(0,0,0,0.07), 0 3px 6px rgba(0,0,0,0.04);
}
.stayery-navcard:hover::before { opacity: 1; }
.stayery-navcard .nc-icon {
    font-size: 1.5rem;
    line-height: 1;
    margin-bottom: 0.55rem;
}
.stayery-navcard .nc-kicker {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: ${caption};
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.stayery-navcard .nc-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: ${ink};
    letter-spacing: -0.01em;
    margin-bottom: 0.35rem;
}
.stayery-navcard .nc-desc {
    font-size: 0.86rem;
    color: ${ink_soft};
    line-height: 1.45;
    margin-bottom: 0.2rem;
}
.stayery-navcard .nc-status {
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    margin-top: 0.6rem;
}
.stayery-navcard .nc-status.ready { background: #e3f5ea; color: #137a3a; }
.stayery-navcard .nc-status.soon  { background: ${bg}; color: ${caption}; border: 1px solid ${border}; }

/* -------------------------------------------------------------------------
   15 — Platzhalter-Banner (für Subpage-Stubs)
   ------------------------------------------------------------------------- */
.stayery-placeholder {
    border: 1px dashed ${muted};
    border-radius: 16px;
    background: ${bg};
    padding: 1.6rem 1.8rem;
    margin: 0.4rem 0 1.2rem;
}
.stayery-placeholder .ph-badge {
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: ${ink};
    background: ${yellow};
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}
.stayery-placeholder .ph-title { font-size: 1.1rem; font-weight: 700; color: ${ink}; margin-bottom: 0.4rem; }
.stayery-placeholder .ph-body { font-size: 0.9rem; color: ${ink_soft}; line-height: 1.5; }
.stayery-placeholder ul { margin: 0.5rem 0 0 1.1rem; padding: 0; }
.stayery-placeholder li { font-size: 0.88rem; color: ${ink_soft}; margin-bottom: 0.25rem; }
</style>
""")


def inject_brand_css() -> None:
    """CSS-Block rendern. Einmal oben auf jeder Page aufrufen."""
    css = _BRAND_CSS.safe_substitute(_brand_tokens())
    st.markdown(css, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    """Editorial-Hero-Block (Eyebrow + H1 + optionaler Subtitle)."""
    sub = f'<div class="stayery-hero-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="stayery-hero">'
        f'<div class="stayery-hero-eyebrow">{eyebrow}</div>'
        f'<h1 style="margin-top:0">{title}</h1>'
        f"{sub}</div>",
        unsafe_allow_html=True,
    )
