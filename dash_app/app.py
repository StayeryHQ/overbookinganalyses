# dash_app/app.py
# Dash application factory for the Stayery overbooking tool.
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
import dash_bootstrap_components as dbc
from dash import Dash, page_container

from dash_app.theme import BLACK, EXTERNAL_STYLESHEETS, YELLOW


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
    title="STAYERY · Overbooking",
)
# WSGI server object (gunicorn dash_app.app:server). Exposed early on purpose.
server = app.server



if __name__ == "__main__":
    # debug=True gives hot-reload locally; the container runs via gunicorn instead.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
