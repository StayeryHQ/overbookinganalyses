# ---------------------------------------------------------------------------
# Public API for the overbooking-analyse project.
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

# Data loading + PII handling + BigQuery access
from .data_loader import (
    PII_COLUMNS,
    average_room_rate_by_property,
    bigquery_healthcheck,
    get_bigquery_client,
    load_clean_reservations,
    load_property_performance,
    load_reservations,
    property_universe,
    strip_pii,
)

# Overbooking decision (newsvendor / cost-optimal allowance)
from .overbooking import (
    DEFAULT_HIGH_DEMAND_MULTIPLIER,
    critical_ratio,
    effective_walk_cost,
    recommend_allowance,
    recommend_from_per_night,
    summarize_property,
)

# Shared feature engineering (train/serve parity)
from .features import (
    add_country_region,
    country_to_region,
    excluded_columns,
    family_feature_lists,
    load_feature_roster,
    log_twin_map,
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

# Daily scoring + the combined data update
from .scoring import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    best_model,
    bucketize,
    cancel_proba,
    list_available_models,
    load_model,
    model_feature_lists,
    refresh_and_score,
    resolve_model,
    score_reservations,
    score_upcoming,
)

# Styling + brand + locations + local-time display
from .utils import (
    apply_stayery_style,
    benchmark_overbooking_allowance,
    categorical_palette,
    color,
    diverging_triplet,
    fmt_ts_local,
    load_brand_config,
    load_risk_buckets,
    load_room_type_capacity,
    local_timezone,
    risk_label,
)

__all__ = [
    # data_loader
    "PII_COLUMNS",
    "average_room_rate_by_property",
    "bigquery_healthcheck",
    "get_bigquery_client",
    "load_clean_reservations",
    "load_property_performance",
    "load_reservations",
    "property_universe",
    "strip_pii",
    # overbooking
    "DEFAULT_HIGH_DEMAND_MULTIPLIER",
    "critical_ratio",
    "effective_walk_cost",
    "recommend_allowance",
    "recommend_from_per_night",
    "summarize_property",
    # features (train/serve parity)
    "add_country_region",
    "country_to_region",
    "model_feature_roster",
    "load_feature_roster",
    "roster_features",
    "excluded_columns",
    "family_feature_lists",
    "log_twin_map",
    # paths
    "configs_dir",
    "data_dir",
    "figures_dir",
    "repo_root",
    "schema_config_path",
    "tables_dir",
    # scoring
    "DEFAULT_MODEL",
    "FALLBACK_MODEL",
    "best_model",
    "bucketize",
    "cancel_proba",
    "list_available_models",
    "load_model",
    "model_feature_lists",
    "refresh_and_score",
    "resolve_model",
    "score_reservations",
    "score_upcoming",
    # utils
    "apply_stayery_style",
    "benchmark_overbooking_allowance",
    "categorical_palette",
    "color",
    "diverging_triplet",
    "fmt_ts_local",
    "load_brand_config",
    "load_risk_buckets",
    "load_room_type_capacity",
    "local_timezone",
    "risk_label",
]
