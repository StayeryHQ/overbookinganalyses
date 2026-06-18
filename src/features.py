# ---------------------------------------------------------------------------
# Shared feature engineering that must be identical in training and serving.
#
# Two responsibilities only:
#   1. Stateless transforms that train AND serve must apply identically
#      (`country_to_region` / `add_country_region`).
#   2. A *generator* + *loader* for the feature roster.
#
# IMPORTANT - where the roster decision lives:
#   The authoritative decision about WHICH columns are model features is made
#   at the END of `notebooks/00_data_audit.ipynb`, based on the full audit, and
#   persisted to `Data/feature_roster.json`. Every notebook and the scoring
#   module LOAD that artifact via `load_feature_roster()` - they do not hardcode
#   feature lists any more (that drift was the root cause of the train/serve
#   mismatch in 08_hazard / score_upcoming).
#
#   `model_feature_roster()` + `NON_FEATURE_COLS` below are only the *mechanism*
#   00 uses to generate the artifact. Treat them as a default; the real,
#   commented blocklist is defined in 00 and passed in explicitly.
#
# Note on `guest_country_region` (corrected 2026-06-18):
#   It is derived from `primaryGuest_address_countryCode`, an ADDRESS field that
#   is frequently only completed at/around check-in - so for not-yet-arrived
#   bookings (exactly what we score) it is missing, and the model would learn
#   "missing => cancel". It is therefore treated as check-in leakage and is NOT
#   a model feature. (An earlier version of this file praised the region
#   encoding's AUC; that benchmark predates the leakage finding and was itself
#   inflated by the leak - do not rely on it.)
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

# Taxonomy ------------------------------------------------------------------
# DACH kept as explicit single-country levels. GB is a large, behaviourally
# distinct market post-Brexit (higher cancel than DACH). Everything else in the
# EU/EEA collapses to EU_other. Long tail (incl. US and all overseas) to
# RoW; missing/blank codes to Unknown. (Region is built for DIAGNOSTICS only -
# it is excluded from the model roster, see module docstring.)

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
# Roster GENERATOR (used once, by 00, to write Data/feature_roster.json)
# ---------------------------------------------------------------------------
# DEFAULT blocklist. The authoritative, commented blocklist lives in
# 00_data_audit.ipynb §11 and is passed into `model_feature_roster(..., non_features=...)`.
# Kept here only as a sensible default / legacy fallback. Anything dropped in
# 00's audit need not be repeated here (it is already gone from the frame).
NON_FEATURE_COLS: Final[frozenset[str]] = frozenset({
    "status", "is_cancelled",                                   # target
    "is_temporal_test", "is_temporal_val", "temporal_split",    # split flags
    "arrival", "departure", "created",                          # raw timestamps
    "company_name_clean", "company_name_combined",              # high-card / intermediate
    COUNTRY_CODE_COL,                                           # raw country code
    # --- collinear duplicates kept in the parquet for reference only ---------
    # gross_amount  : modelled via log_gross_amount (skew-damped); raw is collinear.
    # ratePlan_type : coarse parent of ratePlan_category; the two are nested/collinear.
    "gross_amount", "ratePlan_type",
    # --- check-in / address leakage (missing for not-yet-arrived bookings) ---
    # These are (often) only populated at/around check-in. On upcoming bookings
    # they are blank, so the model would learn "blank => cancel" and explode the
    # forecast. Quantified in 00 §7.5 (and experiments/profile_leakage_quantification).
    "guest_country_region", "primaryGuest_preferredLanguage", "travelPurpose",
    "is_international",
    # --- company leakage --------------------------------------------------------
    # Company / corporate data is frequently entered only at check-in, so the
    # company linkage key is absent at scoring time for upcoming bookings; the
    # history features that depend on it (prior_bookings / prior_cancel_rate)
    # cannot be computed then either. Excluded as a bundle (domain knowledge).
    "has_company", "is_repeat_company", "company_prior_bookings", "company_prior_cancel_rate",
})


def model_feature_roster(df: pd.DataFrame, *, non_features=None, exclude=(), max_card: int = 100):
    """Derive modelable columns from the cleaned frame (00_data_audit).

    Blocklist logic: keep every numeric/bool column and every low-cardinality
    categorical (2..max_card distinct) that is NOT in `non_features` (default
    `NON_FEATURE_COLS`) and NOT in `exclude`. Returns `(numeric, categorical)`.

    NB: because this is blocklist-based, any column you keep in the parquet but
    do NOT want modelled must be listed in `non_features` - otherwise it is
    picked up automatically.
    """
    block = set(NON_FEATURE_COLS if non_features is None else non_features) | set(exclude)
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
        else:
            # dtype-ROBUST categorical detection: anything that is not numeric,
            # bool, or datetime is a categorical candidate. The previous
            # `str(dtype) in {"object","string","category"}` check silently
            # dropped arrow-backed string columns (e.g. "string[pyarrow]",
            # "large_string[pyarrow]"), which produced a DEGENERATE roster
            # (missing property_name / unitGroup_name / guaranteeType / ...).
            if 2 <= int(s.nunique(dropna=True)) <= max_card:
                categorical.append(c)
    return numeric, categorical


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
