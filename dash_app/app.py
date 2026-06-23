# dash_app/app.py
# ---------------------------------------------------------------------------
# The Dash application entry point. This wires up:
#   * the multi-page router (use_pages=True) - every module in dash_app/pages/
#     that calls dash.register_page(...) becomes a route automatically;
#   * the brand Plotly theme (so every chart inherits the Stayery look);
#   * the app shell: a persistent sidebar + a page container that swaps content
#     as you navigate;
#   * a small callback that highlights the active nav link.
#
# Run locally with:  python -m dash_app.app
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Put the repo root (parent of dash_app/) on sys.path before importing src.*.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import dash
from dash import Dash, Input, Output, html, page_container

from src.plotting import apply_plotly_theme

from dash_app.components import sidebar


# ---------------------------------------------------------------------------
# 1) Plotly theme is only applied once here so each figure inherits it.
# ---------------------------------------------------------------------------
apply_plotly_theme()


# ---------------------------------------------------------------------------
# 2) Construct the Dash app with multi-page routing enabled.
#    * __name__               - lets Dash find the assets/ folder next to it.
#    * use_pages=True         - turns on the pages system; Dash auto-discovers
#                               modules under dash_app/pages/ that register_page.
#    * pages_folder           - explicit folder so discovery works regardless of
#                               the working directory at launch.
#    * suppress_callback_exceptions=True - page layouts (and their component ids)
#                               only exist after navigation, so we tell Dash not
#                               to validate callbacks against the initial layout.
#    * title                  - the browser tab title.
# ---------------------------------------------------------------------------
app: Dash = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
    title="Stayery Overbooking Analytics",
)

#expose underlying flask server for WSGI hook
server = app.server

# --- XAI graphs as static pictures------
# files are stored under reports/figures/<model>/.
# page embed via <img src="/figures/<model>/<file>.png">.
# `send_from_directory` streams the file and blocks path traversal outside the
# figures dir (so a crafted URL can't read arbitrary files).
from flask import send_from_directory                       # Flask ships with Dash
_FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"  # <repo>/reports/figures


@server.route("/figures/<path:relpath>")                    # any path below /figures/
def _serve_figure(relpath):                                 # Flask view: stream the PNG
    return send_from_directory(_FIGURES_DIR, relpath)       # 404 automatically if missing


# ---------------------------------------------------------------------------
# 3) The app SHELL layout. This frame is constant across pages; only the
#    `page_container` content changes as the user navigates.
#      * the sidebar (built from dash.page_registry) sits on the left;
#      * page_container renders the active page's `layout` on the right.
#    The two are placed in a CSS grid (.stayery-app) defined in stayery.css.
# ---------------------------------------------------------------------------
app.layout = html.Div(
    [
        # Left column: persistent navigation.
        sidebar(),
        # Right column: the routed page content goes here.
        html.Div(page_container, className="stayery-content"),
    ],
    className="stayery-app",
)


# ---------------------------------------------------------------------------
# 4) Active-nav-link highlighting. The sidebar builds one dcc.Link per page with
#    a pattern-matching id {"type": "nav-link", "path": <route>}. This callback
#    listens to the current URL (the built-in `_pages_location` component that
#    use_pages adds) and recomputes each link's className so the matching one
#    gets the yellow "active" pill.
#
#    Dash callback primer:
#      * @app.callback declares a reactive function.
#      * Output(...) = what it writes; here, the `className` of EVERY nav link
#        (ALL matches the pattern-matching id).
#      * Input(...)  = what triggers it; here, the URL pathname.
#      * The function returns a list aligned to the matched outputs.
# ---------------------------------------------------------------------------
@app.callback(
    # Write the className of every component whose id matches the nav-link pattern.
    Output({"type": "nav-link", "path": dash.ALL}, "className"),
    # Trigger on URL changes. use_pages injects a dcc.Location with this id.
    Input("_pages_location", "pathname"),
)
def _highlight_active_link(pathname: str):
    """Return the className list for all nav links, marking the active one."""
    # dash.callback_context tells us which outputs we must return (one per link).
    # We read the matched output ids to recover each link's path.
    outputs = dash.callback_context.outputs_list
    classes = []
    for out in outputs:
        # The link's route is stored in its id under the "path" key.
        link_path = out["id"]["path"]
        # Home is "/"; other pages match when the URL starts with their path.
        is_active = (pathname == link_path) or (
            link_path != "/" and (pathname or "").startswith(link_path)
        )
        # Append the active modifier class when this link is the current page.
        classes.append("stayery-navlink stayery-navlink--active" if is_active
                       else "stayery-navlink")
    return classes


# ---------------------------------------------------------------------------
# 4b) Per-page SIDEBAR FILTERS. Each page registers a filter-controls builder in
#     dash_app.filters_registry (at import). This callback renders the matching
#     page's filters into the sidebar slot whenever the URL changes - so filters
#     live in the sidebar AND vary per page, with no app<->page coupling.
# ---------------------------------------------------------------------------
from dash_app import filters_registry  # noqa: E402 - local import keeps app top tidy


@app.callback(
    Output("sidebar-page-filters", "children"),   # the slot added in components/sidebar.py
    Input("_pages_location", "pathname"),          # current route
)
def _render_sidebar_filters(pathname: str):
    """Drop the active page's filter controls into the sidebar slot (or nothing)."""
    return filters_registry.render(pathname)


# ---------------------------------------------------------------------------
# 5) Server launch - ONLY under __main__, so `import dash_app.app` never starts
#    a server (keeps the import import-clean for tests / tooling).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True gives hot-reload + in-browser error pages during development.
    app.run(debug=True, port=8050)
