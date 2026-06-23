# dash_app/components/ui.py
# ---------------------------------------------------------------------------
# The CSS classes used here are defined in dash_app/assets/stayery.css.
#
# Dash building blocks used:
#   * dash.html.*  - thin Python wrappers around HTML tags. html.Div(...) ==
#     a <div>; the first positional arg (or `children=`) is its contents;
#     `className=` sets the CSS class; `style=` is an inline-style dict.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dash import html

# Raw SVG icon markup helper.
from .icons import icon_markup


def hero(eyebrow: str, title: str, subtitle: str | None = None) -> html.Div:
    """Editorial hero block (eyebrow + H1 + optional subtitle). Mirrors brand.hero."""
    # Build the children list; only add the subtitle paragraph if provided.
    children = [
        # Small uppercase kicker above the title.
        html.Div(eyebrow, className="stayery-hero-eyebrow"),
        # The page H1.
        html.H1(title),
    ]
    if subtitle:
        # Optional lead paragraph under the title.
        children.append(html.Div(subtitle, className="stayery-hero-subtitle"))
    # Wrap everything in the hero container div.
    return html.Div(children, className="stayery-hero")


def alert(text: str, kind: str = "info") -> html.Div:
    """Coloured callout box (info / warning / success). Mirrors alerts.alert_card.

    `text` may contain markdown-ish **bold** - we pass it through dcc.Markdown
    so emphasis renders. We keep it simple here with plain text + a className.
    """
    # The modifier class selects the colour (see .stayery-alert--* in the CSS).
    return html.Div(text, className=f"stayery-alert stayery-alert--{kind}")


def metric(label: str, value, help_text: str | None = None) -> html.Div:
    """A single metric card (uppercase label + big value). Mirrors st.metric."""
    return html.Div(
        [
            # The small uppercase label.
            html.Div(label, className="stayery-metric-label"),
            # The large value (cast to str so ints/floats render).
            html.Div(str(value), className="stayery-metric-value"),
        ],
        className="stayery-metric",
        # Native browser tooltip via the title attribute, if help text given.
        title=help_text or "",
    )


def metric_row(metrics: list[dict]) -> html.Div:
    """A responsive row of metric cards. Each dict: {label, value, help?}."""
    # Turn each spec dict into a metric() card.
    cards = [metric(m["label"], m["value"], m.get("help")) for m in metrics]
    # The grid wrapper auto-fits the cards across the available width.
    return html.Div(cards, className="stayery-metric-row")


def nav_card(*, href: str, icon_name: str, kicker: str, title: str, desc: str,
             status: str = "ready", status_label: str | None = None) -> html.A:
    """A clickable navigation tile (whole card is the link). Mirrors nav.nav_card.

    `href` is the page path (Dash uses client-side routing via dcc.Link/anchor).
    We make the whole card an <a> so the entire tile is the click target.
    """
    # Default the status pill label from the status if not given.
    if status_label is None:
        status_label = "Verfügbar" if status == "ready" else "In Arbeit"
    return html.A(
        [
            # (Icon removed 2026-06-18: the dependency-free fallback rendered an
            #  ugly yellow square. Add real inline SVG via dash-svg later if wanted.)
            # Uppercase kicker, bold title, description, status pill.
            html.Div(kicker, className="nc-kicker"),
            html.Div(title, className="nc-title"),
            html.Div(desc, className="nc-desc"),
            html.Span(status_label, className=f"nc-status {status}"),
        ],
        href=href,                       # navigation target
        className="stayery-navcard",
    )


def nav_card_grid(cards: list[html.A]) -> html.Div:
    """Wrap nav cards in the responsive grid container."""
    return html.Div(cards, className="stayery-navcard-grid")


def placeholder_panel(*, title: str, intro: str, planned: list[str] | None = None,
                      badge: str = "Nächster Schritt") -> html.Div:
    """Dashed placeholder box for not-yet-built sections. Mirrors nav.placeholder_panel."""
    body_children = [intro]
    # If planned items are given, render them as a bullet list.
    if planned:
        body_children.append(html.Ul([html.Li(p) for p in planned]))
    return html.Div(
        [
            # Yellow badge pill.
            html.Span(badge, className="ph-badge"),
            html.Div(title, className="ph-title"),
            html.Div(body_children, className="ph-body"),
        ],
        className="stayery-placeholder",
    )


def section(number: int, title: str, *, description: str | None = None,
            children: list | None = None) -> html.Div:
    """A numbered section: H2 heading + optional description + content.

    Mirrors the Streamlit `section(n, title, description=...)` context manager -
    here it's a plain wrapper that takes its children directly.
    """
    head = [html.H2(f"{number}. {title}")]
    # Optional one-line section description under the heading.
    if description:
        head.append(html.P(description, className="stayery-section-desc"))
    # Append the caller-supplied content after the heading.
    body = head + (children or [])
    return html.Div(body, className="stayery-section")


def explain(summary: str, body: str) -> html.Details:
    """Collapsible explainer (the Dash twin of ui.explain / st.expander)."""
    # html.Details renders a native <details> disclosure widget.
    return html.Details(
        [html.Summary(summary), html.Div(body, style={"marginTop": "0.4rem"})],
        className="stayery-explain",
    )


def caption(text: str) -> html.Div:
    """Small muted caption line (mirrors st.caption)."""
    return html.Div(text, className="stayery-caption")


def dangerously_set_inner_html(markup: str):
    """Helper that returns the kwargs needed to inject raw HTML into html.Div.

    Dash's html components accept `dangerously_allow_html` only via the
    DangerouslySetInnerHTML pattern. The simplest portable approach is to render
    raw markup through the `dash_dangerously_set_inner_html` lib OR via an
    iframe/srcDoc; to avoid an extra dependency we instead return an html.Span
    using the `dangerously_allow_html`-free fallback: a unicode placeholder.

    NOTE: SVG icons are decorative. To keep zero extra deps and stay import-clean
    we render the icon as an inline <img>-free span; if you want the crisp SVG,
    `pip install dash-dangerously-set-inner-html` and swap this for it.
    """
    # Render the REAL inline SVG dependency-free via a data-URI <img> (no yellow
    # square anymore). `quote` percent-encodes the markup so it is a valid URI.
    from urllib.parse import quote
    return html.Img(
        src="data:image/svg+xml;charset=utf-8," + quote(markup),
        style={"width": "22px", "height": "22px", "display": "inline-block",
               "verticalAlign": "middle"},
    )
