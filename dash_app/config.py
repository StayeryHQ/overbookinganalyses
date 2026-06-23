# dash_app/config.py
# ---------------------------------------------------------------------------
# Central configuration for the Dash app: repo paths, the page registry order,
# the backend mode, and the model selection. Keeping these in ONE place is what
# makes the app scalable — adding a model or flipping the backend is a one-line
# change here, and every page reads from this module.
# ---------------------------------------------------------------------------

# `from __future__ import annotations` makes all type hints lazy strings, so we
# can write modern hints (str | None) even on Python 3.10.
from __future__ import annotations

# `os` lets us read environment variables (the OVERBOOKING_BACKEND override).
import os
# `Path` is the object-oriented filesystem path API we use for repo paths.
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo paths. This file lives at <repo>/dash_app/config.py, so the repo root is
# the parent of the `dash_app` directory (one level up).
# ---------------------------------------------------------------------------
# `Path(__file__).resolve()` = absolute path to THIS file; `.parents[1]` walks
# up two levels (config.py -> dash_app/ -> <repo root>).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
# The `Data/` directory holds the dummy snapshot JSON + (later) the real
# scored parquet. We never write outside the repo.
DATA_DIR: Path = REPO_ROOT / "Data"
# Where the trained model cards live (reports/tables/<NN_model>/model_card.json).
TABLES_DIR: Path = REPO_ROOT / "reports" / "tables"

# ---------------------------------------------------------------------------
# Backend mode — 'dummy' (synthetic, no model needed) or 'real' (src.scoring).
# Mirrors streamlit_app/backend/__init__.py: env override OVERBOOKING_BACKEND,
# default 'dummy' so the app renders with no trained models on disk.
# ---------------------------------------------------------------------------
def default_backend_mode() -> str:
    """Return the startup backend mode from the environment (default 'dummy')."""
    # `os.environ.get` reads the variable; we lower-case + strip and fall back
    # to 'dummy' if it's unset or empty.
    mode = os.environ.get("OVERBOOKING_BACKEND", "dummy").strip().lower()
    # Only 'real' is accepted as an alternative; anything else => 'dummy'.
    return "real" if mode == "real" else "dummy"


# ---------------------------------------------------------------------------
# Model registry mirror. The REAL backend ultimately defers to
# src.scoring.MODEL_REGISTRY / list_available_models(), but for the model
# dropdown we want a friendly label per registry name. Adding a model = one
# new entry here AND in src.scoring (no page/UI change required).
# ---------------------------------------------------------------------------
MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistische Regression (01)",   # baseline linear model
    "xgboost": "XGBoost (02)",                  # gradient-boosted trees
    "histgb": "HistGradientBoosting (03)",      # sklearn boosted trees
}

# Fallback model name used when the dropdown has not been touched yet. None =
# let src.scoring.best_model() pick the model (highest AP among well-calibrated
# models — see the Brier gate in src.scoring).
DEFAULT_MODEL: str | None = None

# How far ahead (days) the dummy snapshot is generated / the window spans.
# Mirrors GENERATION_HORIZON_DAYS in the Streamlit backend.
GENERATION_HORIZON_DAYS: int = 35

# ---------------------------------------------------------------------------
# Page registry order. dash.register_page assigns an `order` so the sidebar
# lists pages in this sequence. Centralised here so reordering/adding a page is
# a single edit. The integers match the German titles' historical numbering.
# ---------------------------------------------------------------------------
PAGE_ORDER: dict[str, int] = {
    "home": 0,                       # Übersicht / landing
    "auslastung": 1,                 # Occupancy & arrivals (STUB)
    "overbooking_predictions": 2,    # Overbooking Predictions (FULL)
    "datenaktualisierung": 3,        # Datenaktualisierung (STUB)
    "modell_performance": 4,         # Model & performance (FULL)
}
