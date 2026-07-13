# ---------------------------------------------------------------------------
# Shared feature engineering — identical in training and serving.
#
# Two jobs: (1) stateless transforms both sides apply the same way
# (country -> region), (2) generator + loader for the feature roster.
#
# WHICH columns are model features is decided once, at the end of notebook 00,
# and persisted to Data/feature_roster.json — everything loads that artifact.
# FEATURE_EXCLUSIONS below is the single list of what is NOT a feature and why.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Column names
COUNTRY_CODE_COL: Final[str] = "primaryGuest_address_countryCode"
REGION_COL:       Final[str] = "guest_country_region"

# Region taxonomy: DE/AT/CH explicit, GB separate (distinct market, cancels more
# than DACH), rest of EU/EEA -> EU_other, everything else -> RoW, blank -> Unknown.
# Diagnostics only — the region column is excluded from the model roster.

DACH: Final[frozenset[str]] = frozenset({"DE", "AT", "CH"})
EU_EEA: Final[frozenset[str]] = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE", "IS", "LI", "NO",
})
# The full set of levels this function can emit
REGION_LEVELS: Final[tuple[str, ...]] = ("DE", "AT", "CH", "GB", "EU_other", "RoW", "Unknown")

_MISSING_TOKENS: Final[frozenset[str]] = frozenset({"", "<NA>", "NAN", "NONE", "NULL", " "})


def country_to_region(code) -> str:
    """Map a single ISO-2 country code to its region level. NaN/blank -> Unknown.

    Tests missingness before any equality, because `pd.NA == "x"`
    returns NA and `bool(NA)` raises ("boolean value of NA is ambiguous").
    """
    if code is None or pd.isna(code):
        return "Unknown"
    c = str(code).strip().upper()
    if c in _MISSING_TOKENS:
        return "Unknown"
    if c in DACH:
        return c              # DE / AT / CH explicit
    if c == "GB":
        return "GB"
    if c in EU_EEA:
        return "EU_other"
    return "RoW"


def add_country_region(
    df: pd.DataFrame,
    *,
    country_col: str = COUNTRY_CODE_COL,
    out_col: str = REGION_COL,
) -> pd.DataFrame:
    """Add the `guest_country_region` column to `df` (mutates and returns it).

    Vectorised and dtype-robust: emits a plain `object` column
    so downstream SimpleImputer / OneHotEncoder never trip over
    pandas nullable missing values. If the country column is absent, every row
    is labelled "Unknown".
    """
    if country_col not in df.columns:
        df[out_col] = pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")
        return df

    codes = (df[country_col].astype("string").str.strip().str.upper()
             .fillna("").to_numpy(dtype=object))
    region = np.full(len(df), "RoW", dtype=object)
    region[np.isin(codes, list(EU_EEA))] = "EU_other"
    region[codes == "GB"] = "GB"
    for c in DACH:
        region[codes == c] = c          # override EU_other for DACH
    region[np.isin(codes, list(_MISSING_TOKENS))] = "Unknown"

    df[out_col] = pd.Series(region, index=df.index, dtype="object")
    return df


# ---------------------------------------------------------------------------
# Roster GENERATOR (used by 00 to write Data/feature_roster.json)
# ---------------------------------------------------------------------------
# Blocklist logic: every clean-parquet column becomes a model feature UNLESS it
# is listed here, grouped by reason. To keep a new column out, add it to the
# right group. The roster's `excluded` audit map is derived from this dict.
FEATURE_EXCLUSIONS: Final[dict[str, list[str]]] = {
    "target or split metadata (never a feature)": [
        "status", "is_cancelled", "is_canceled_by_arrival",
        "is_temporal_test", "is_temporal_val", "temporal_split",
        "cancel_days_before_arrival", "outcome_known_date",
    ],
    "raw timestamp (engineered into lead-time / calendar features)": [
        "arrival", "departure", "created",
    ],
    "high-cardinality key / intermediate": [
        "company_name_clean", "company_name_combined", COUNTRY_CODE_COL,
    ],
    "collinear duplicate (modelled via its skew-damped / finer twin)": [
        "gross_amount",    # -> log_gross_amount
        "ratePlan_type",   # -> ratePlan_category (coarse parent)
    ],
    "check-in / address leakage (blank for not-yet-arrived bookings)": [
        "guest_country_region", "primaryGuest_preferredLanguage",
        "travelPurpose", "is_international",
    ],
    "company leakage (linkage key absent at scoring time)": [
        "has_company", "is_repeat_company",
        "company_prior_bookings", "company_prior_cancel_rate",
    ],
    "dynamic point-in-time feature (built per scoring-date in build_features)": [
        "days_until_arrival", "days_since_booking",
        "pct_lead_time_elapsed", "is_within_7d_of_arrival",
    ],
}


def excluded_columns() -> dict[str, str]:
    """Flatten FEATURE_EXCLUSIONS to `{column: reason}` — the roster audit trail."""
    return {col: reason for reason, cols in FEATURE_EXCLUSIONS.items() for col in cols}


def model_feature_roster(df: pd.DataFrame, *, non_features=None, exclude=(), max_card: int = 100):
    """Split the cleaned frame into `(numeric, categorical)` MODEL features.

    A column is kept unless it is in the exclusion taxonomy (`excluded_columns()`,
    overridable via `non_features`) or in `exclude`. numeric/bool -> numeric;
    low-cardinality (2..max_card) object/string/category -> categorical;
    timestamps/durations and high-cardinality categoricals are skipped.
    """
    block = set(excluded_columns() if non_features is None else non_features) | set(exclude)
    numeric, categorical = [], []
    for c in df.columns:
        if c in block:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        elif (pd.api.types.is_datetime64_any_dtype(s)
              or pd.api.types.is_timedelta64_dtype(s)):
            continue  # timestamps / durations are never model features
        elif 2 <= int(s.nunique(dropna=True)) <= max_card:
            categorical.append(c)
    return numeric, categorical


# ---------------------------------------------------------------------------
# One roster, model-FAMILY-aware views (decided 2026-06-30)
# ---------------------------------------------------------------------------
# We keep a single candidate roster, but a skewed numeric column and its `_log`
# twin both live in the parquet. Linear models want the log twin (scaling /
# linearity); tree models want the raw column and ignore the resulting
# collinearity. These helpers derive the per-family list from the one roster so
# we never maintain three rosters.
def log_twin_map(numeric: list[str]) -> dict[str, str]:
    """`{raw: raw_log}` for every numeric column whose `<name>_log` partner is
    also in `numeric` (e.g. `los_nights` -> `los_nights_log`). Standalone logs
    with no raw partner in the roster (e.g. `log_gross_amount`) are not listed."""
    present = set(numeric)
    return {c: f"{c}_log" for c in numeric if f"{c}_log" in present}


def family_feature_lists(roster: dict, family: str) -> tuple[list[str], list[str]]:
    """`(numeric, categorical)` tuned for a model FAMILY from one roster dict.

        family='tree'   -> raw skewed columns; drop their `_log` twins.
        family='linear' -> use the `_log` twin in place of each raw skewed column.

    `roster` is the loaded feature_roster.json (must carry `log_twins`, written
    by 00 §11). Categoricals are family-agnostic.
    """
    num = list(roster["numeric"])
    cat = list(roster["categorical"])
    twins = roster.get("log_twins", {})          # {raw: raw_log}
    logs = set(twins.values())
    if family == "tree":
        num = [c for c in num if c not in logs]                  # keep raw, drop logs
    elif family == "linear":
        num = [twins.get(c, c) for c in num if c not in logs]    # raw -> log twin
    else:
        raise ValueError("family must be 'tree' or 'linear'")
    return num, cat


# ---------------------------------------------------------------------------
# Roster LOADER (used by every notebook + scoring.py - the single source)
# ---------------------------------------------------------------------------
ROSTER_FILENAME: Final[str] = "feature_roster.json"


def _default_roster_path() -> Path:
    # Imported lazily so importing this module never hard-depends on paths.
    from .paths import data_dir
    return data_dir() / ROSTER_FILENAME


def load_feature_roster(path: str | Path | None = None) -> dict:
    """Load the roster artifact written by 00_data_audit.ipynb §11.

    Returns the full dict, e.g.:
        {"numeric": [...], "categorical": [...], "dynamic_numeric": [...],
         "target": "status", "excluded": {...}, "generated_at_utc": "...", ...}

    Raises a clear error if the artifact is missing (run 00 first).
    """
    p = Path(path) if path is not None else _default_roster_path()
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run notebooks/00_data_audit.ipynb (§11 writes the "
            f"feature roster) before loading features."
        )
    with p.open("r") as fh:
        return json.load(fh)


def roster_features(path: str | Path | None = None, *, include_dynamic: bool = False):
    """Convenience: return `(numeric, categorical)` from the roster artifact.

    `include_dynamic=True` appends the dynamic (scoring-time) numeric features -
    use this in the hazard notebook (08), which models static + dynamic.
    """
    r = load_feature_roster(path)
    numeric = list(r.get("numeric", []))
    if include_dynamic:
        numeric = numeric + list(r.get("dynamic_numeric", []))
    return numeric, list(r.get("categorical", []))
