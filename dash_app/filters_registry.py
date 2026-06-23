# dash_app/filters_registry.py
# ---------------------------------------------------------------------------
# Tiny registry that lets each PAGE contribute its own filter controls to the
# shared sidebar slot (#sidebar-page-filters). A page calls register(path, fn)
# at import time; app.py's path callback calls render(pathname) to drop the
# matching page's filters into the sidebar. This keeps filters IN the sidebar
# and VARIABLE PER PAGE, without app.py needing to import every page.
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable

# path -> a zero-arg function that returns the page's filter controls (a list of
# Dash components). Populated when the page modules are imported (use_pages).
_REGISTRY: dict[str, Callable[[], list]] = {}


def register(path: str, builder: Callable[[], list]) -> None:
    """Register a page's filter-controls builder under its route `path`."""
    _REGISTRY[path] = builder


def render(pathname: str | None):
    """Return the filter controls for `pathname` (or an empty list if none)."""
    builder = _REGISTRY.get(pathname or "")
    # Call the builder fresh each navigation so option lists reflect current data.
    return builder() if builder is not None else []
