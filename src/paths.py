# ---------------------------------------------------------------------------
# Project path resolution.
#
# WHY this exists:
#   - Notebooks live in OverbookingAnalyse/notebooks/
#   - Source code lives in OverbookingAnalyse/src/
#   - Configs live in OverbookingAnalyse/configs/
#   - Cached parquet data lives in OverbookingAnalyse/Data/
#   - Plots & tables land in OverbookingAnalyse/reports/
# Anchoring everything to a single "repo root" (the directory that contains
# `pyproject.toml`) means notebooks and scripts work no matter where they are
# ---------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# ---- Repo root discovery --------------------------------------------------

_MARKERS = ("pyproject.toml", ".git")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Walk upward from this file until we find pyproject.toml or .git.

    Returns the project root directory. Cached so repeated calls are free.
    Raises if it cannot be found (running outside a real checkout).
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if any((parent / m).exists() for m in _MARKERS):
            # Make sure we are at the *project* root, not just any .git folder.
            if (parent / "pyproject.toml").exists():
                return parent
    raise RuntimeError(
        "Could not locate project root (no pyproject.toml found upwards from "
        f"{here}). Are you running outside the repo checkout?"
    )


# ---- Convenience getters --------------------------------------------------

def configs_dir() -> Path:
    return repo_root() / "configs"


def _paths_yaml() -> dict[str, Any]:
    """Load configs/paths.yaml once and cache it."""
    with (configs_dir() / "paths.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_dir() -> Path:
    """Where raw + cached parquet data lives. Created lazily."""
    p = repo_root() / _paths_yaml().get("raw_data_dir", "Data")
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    p = repo_root() / _paths_yaml().get("figures_dir", "reports/figures")
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir() -> Path:
    p = repo_root() / _paths_yaml().get("tables_dir", "reports/tables")
    p.mkdir(parents=True, exist_ok=True)
    return p


def brand_config_path() -> Path:
    """Path to configs/stayery_brand.yaml."""
    return configs_dir() / "stayery_brand.yaml"


def locations_config_path() -> Path:
    """Path to configs/locations.yaml."""
    return configs_dir() / "locations.yaml"


def schema_config_path() -> Path:
    """Path to configs/reservations_schema.json."""
    return configs_dir() / "reservations_schema.json"
