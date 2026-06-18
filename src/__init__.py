# ---------------------------------------------------------------------------
# Public API for the overbooking-analyse project.
#
#
# Conventions:
#   * Loading + caching + PII strip       → data_loader
#   * Brand styling + colours + locations → utils
#   * Path resolution                     → paths
#   * Daily scoring + risk bucketing      → scoring
#   * Cleaning, feature engineering,
#     model training, evaluation, XAI    → live in notebooks and will transfer later
# ---------------------------------------------------------------------------

from __future__ import annotations

# Data loading + PII handling
from .data_loader import (
    PII_COLUMNS,
    load_clean_reservations,
    load_reservations,
    strip_pii,
)

# Shared feature engineering (train/serve parity)
from .features import (
    add_country_region,
    country_to_region,
    load_feature_roster,
    model_feature_roster,
    roster_features,
)

# Path resolution
from .paths import (
    configs_dir,
    data_dir,
    figures_dir,
    repo_root,
    schema_config_path,
    tables_dir,
)

# Daily scoring
from .scoring import (
    HIGH_THR,
    LOW_THR,
    best_model_by_auc,
    bucketize,
    list_available_models,
    load_model,
    model_feature_lists,
    score_upcoming,
)

# Styling + brand + locations
from .utils import (
    apply_stayery_style,
    benchmark_overbooking_allowance,
    categorical_palette,
    color,
    diverging_triplet,
    load_brand_config,
    load_locations,
)

__all__ = [
    # data_loader
    "PII_COLUMNS", "load_clean_reservations",
    "load_reservations", "strip_pii",
    # features (train/serve parity)
    "add_country_region", "country_to_region", "model_feature_roster",
    "load_feature_roster", "roster_features",
    # paths
    "configs_dir", "data_dir", "figures_dir", "repo_root",
    "schema_config_path", "tables_dir",
    # scoring
    "HIGH_THR", "LOW_THR", "best_model_by_auc", "bucketize",
    "list_available_models", "load_model", "model_feature_lists", "score_upcoming",
    # utils
    "apply_stayery_style", "benchmark_overbooking_allowance",
    "categorical_palette", "color", "diverging_triplet",
    "load_brand_config", "load_locations",
]
