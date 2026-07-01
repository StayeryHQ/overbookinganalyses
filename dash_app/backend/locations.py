# dash_app/backend/locations.py

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_locations() -> list[tuple[str, str, int]]:
    """Deprecated placeholder until the SQL-backed property table is wired in."""
    return []
