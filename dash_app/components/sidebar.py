# dash_app/components/sidebar.py
# ---------------------------------------------------------------------------
# The persistent left navigation. multi-page apps, every registered page appears in `dash.page_registry`
# adding a page automatically adds a navigation
# ---------------------------------------------------------------------------

from __future__ import annotations


import dash
from dash import dcc, html


def sidebar() -> html.Div:
    """Build the sidebar: brand block + a nav link per registered page."""
    # Collect (path, name, order) for every registered page. page_registry is a
    # dict keyed by module name; each value has 'path', 'name', 'order', etc.
    pages = list(dash.page_registry.values())
    pages.sort(key=lambda p: (p.get("order") if p.get("order") is not None else 999))

    links = []
    for p in pages:
        links.append(
            dcc.Link(
                # The visible label = the page's registered name (German title).
                p["name"],
                # The route this link points to.
                href=p["path"],
                # Base nav-link styling; the active state is added in a callback.
                className="stayery-navlink",
                # An id so the active-state callback can target each link.
                id={"type": "nav-link", "path": p["path"]},
            )
        )

    return html.Div(
        [
            # Brand header block.
            html.Div("STAYERY", className="stayery-sidebar-brand"),
            html.Div("Cancellation Analytics", className="stayery-sidebar-sub"),
            # Section heading above the nav links.
            html.Div("Pages", className="stayery-sidebar-heading"),
            # The generated nav links.
            html.Div(links),
            # Per-page filter slot: a callback in app.py fills this with the
            # current page's filters (from dash_app.filters_registry).
            html.Div(id="sidebar-page-filters", className="stayery-sidebar-filters"),
        ],
        className="stayery-sidebar",
    )
