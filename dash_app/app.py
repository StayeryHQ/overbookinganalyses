# dash_app/app.py
#
# Run locally:   uv run python -m dash_app.app
# WSGI (prod):   gunicorn dash_app.app:server        (server is exposed below)
#
# Background callbacks use the environment-driven manager the Dash docs recommend:
#   * no REDIS_URL  -> DiskcacheManager (local dev)               [default]
#   * REDIS_URL set -> CeleryManager backed by Redis (future prod)
# so the SAME code works in both contexts with no edits (see project instructions).

from __future__ import annotations

import os

import dash
import dash_mantine_components as dmc
from dash import Dash, _dash_renderer, dcc, html, page_container

# dash-mantine-components 2.x renders against React 18. Dash 4.3.0 already ships
# React 18 (default 18.3.1); we pin 18.2.0 explicitly
# NOTE: must run BEFORE the Dash() instance is created.
_dash_renderer._set_react_version("18.2.0")

from dash_app.theme import DMC_THEME, EXTERNAL_STYLESHEETS


def _background_manager():
    """DiskCache locally; Celery+Redis when REDIS_URL is set. Verified against the
    Dash Background Callbacks docs (DiskcacheManager / CeleryManager, background=True)."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        from celery import Celery  # optional dep; only needed in the Redis/prod path
        from dash import CeleryManager
        celery_app = Celery(__name__, broker=redis_url, backend=redis_url)
        return CeleryManager(celery_app)
    import diskcache
    from dash import DiskcacheManager
    cache_dir = os.environ.get("DASH_CACHE_DIR", ".dash_cache")
    return DiskcacheManager(diskcache.Cache(cache_dir))


app = Dash(
    __name__,
    use_pages=True,
    background_callback_manager=_background_manager(),
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,   # callbacks target components on page-scoped layouts
    title="STAYERY · Cancellation Analytics",
    # Disable Dash's default "Updating..." browser-tab title. The pages poll a job file on
    # a short dcc.Interval, which otherwise makes that title flicker permanently even when
    # nothing is running. In-page progress rings (ui.job_loader) already show real progress.
    update_title=None,
)

_original_strip = app.strip_relative_path
def _patched_strip(path):
    if path is None:
        return None
    prefix = app.config.requests_pathname_prefix
    if prefix == "/":
        return _original_strip(path)
    if path.startswith(prefix.rstrip("/")):
        return _original_strip(path)
    # Renderer already stripped the prefix — just remove slashes
    return path.strip("/")
app.strip_relative_path = _patched_strip


# WSGI server object (gunicorn dash_app.app:server). Exposed early on purpose.
server = app.server

# Public URL prefix: empty (/) locally, /_carogician/ behind Traefik.
# Used for html.A hrefs since dcc.Link's client-side navigation double-strips
# the prefix and breaks Dash's internal pages callback.
_URL_PREFIX = os.environ.get("DASH_URL_BASE_PATHNAME", "/")


def _navbar() -> html.Header:
    """Top navigation shell. The links live in an id'd container and are rendered
    by the callback below, which reads dcc.Location - that is how the active link
    gets its `.active` class now that dbc.NavLink is gone (styling: brand.css)."""
    # html.A (not dcc.Link) — dcc.Link's client-side navigation sets
    # _pages_location.pathname without the prefix, which then causes Dash's
    # strip_relative_path to error (double-strip). html.A forces a full page
    # load so pathname comes from window.location.pathname (with prefix).
    # _URL_PREFIX is DASH_URL_BASE_PATHNAME (/_carogician/) or "/" locally.
    brand = html.A(
        html.Div([
            html.Span(className="stayery-accent"),
            html.Span("STAYERY", className="stayery-wordmark"),
            html.Span("Cancellation Analytics", className="stayery-subbrand"),
        ], className="stayery-brand-inner"),
        href=_URL_PREFIX, className="stayery-brand")
    return html.Header(
        html.Div([brand, html.Nav(id="stayery-nav", className="stayery-nav")],
                 className="stayery-header-inner"),
        className="stayery-header")


@dash.callback(dash.Output("stayery-nav", "children"), dash.Input("app-url", "pathname"))
def _nav_links(pathname):
    """Re-render the nav links on every route change; the current page gets
    `.active` (underline, see brand.css)."""
    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
    # html.A (not dcc.Link) — same reason as the brand link above.
    # p["relative_path"] already includes the prefix (or "/" locally).
    return [html.A(p["name"], href=p["relative_path"],
                    className="stayery-navlink"
                    + (" active" if pathname == p["relative_path"] else ""))
            for p in pages]



app.layout = dmc.MantineProvider(
    html.Div(
        [dcc.Location(id="app-url"),
         dcc.Store(id="cost-store", storage_type="local"), _navbar(), page_container],
        style={"maxWidth": "1500px", "margin": "0 auto", "padding": "0 12px"}),
    theme=DMC_THEME,
    forceColorScheme="light",
)


if __name__ == "__main__":
    # debug=True gives hot-reload locally; the container runs via gunicorn instead.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
