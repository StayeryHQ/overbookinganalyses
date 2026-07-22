# ---------------------------------------------------------------------------
# Shared feature engineering  identical in training and serving.
#
# Two jobs: (1) stateless transforms both sides apply the same way
# (country -> region), (2) generator + loader for the feature roster.
#
# WHICH columns are model features is decided once, at the end of notebook 00,
# and persisted to Data/feature_roster.json  everything loads that artifact.
# FEATURE_EXCLUSIONS below is the single list of what is NOT a feature and why.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Column names
COUNTRY_CODE_COL: Final[str] = "primaryGuest_address_countryCode"
REGION_COL:       Final[str] = "guest_country_region"

# Region taxonomy: DE/AT/CH explicit, GB separate (distinct market, cancels more
# than DACH), rest of EU/EEA -> EU_other, everything else -> RoW, blank -> Unknown.
# Diagnostics only  the region column is excluded from the model roster.

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
    """Flatten FEATURE_EXCLUSIONS to `{column: reason}`  the roster audit trail."""
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


# The ONE model -> family map (single source of truth). Every place a model is fit or
# scored resolves its columns through `model_feature_lists()` below, so training,
# hazard, scoring, eval and the dashboard can never drift onto different feature sets.
MODEL_FAMILY: Final[dict[str, str]] = {
    "logreg": "linear", "xgboost": "tree", "histgb": "tree", "hazard": "tree",
}


def model_family(model_name: str) -> str:
    """Feature FAMILY ('linear' | 'tree') for a model name - the single mapping."""
    try:
        return MODEL_FAMILY[model_name]
    except KeyError:
        raise KeyError(f"unknown model {model_name!r}; known: {list(MODEL_FAMILY)}") from None


def model_feature_lists(model_name: str, *, roster: dict | None = None,
                        present_in=None) -> tuple[list[str], list[str]]:
    """THE single source of truth for the columns a model CONSUMES.

    Derived from the one roster + the model's family (tree -> raw skewed columns;
    linear -> their `_log` twins; categoricals are family-agnostic). Pass `present_in`
    (e.g. a frame's `.columns`) to keep only columns that actually exist there.

    Use this EVERYWHERE a model is fit or scored. Do NOT use the raw-roster superset
    (`roster_features`) for that - it carries both raw AND `_log` twins and is only a
    presence guard / null-audit helper.
    """
    r = roster if roster is not None else load_feature_roster()
    num, cat = family_feature_lists(r, model_family(model_name))
    if present_in is not None:
        cols = set(present_in)
        num = [c for c in num if c in cols]
        cat = [c for c in cat if c in cols]
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


# ---------------------------------------------------------------------------
# Roster GENERATOR at RUNTIME (the src twin of notebook 00 §11)
# ---------------------------------------------------------------------------
# WHY: previously the roster (Data/feature_roster.json) was ONLY written by
# notebook 00. A fresh deploy / rebuilt cache therefore had no roster, and
# load_feature_roster() -> FileNotFoundError broke build_features / scoring /
# retraining with no way to self-heal. These helpers let the runtime regenerate
# the exact same artifact (verified byte-for-byte against the notebook's output
# for numeric / categorical / log_twins / excluded / ratePlan_category_map).

# Dynamic (scoring-time) features: point-in-time per (booking, scoring-date),
# built in src.scoring.build_features. NEVER static roster features.
DYNAMIC_NUMERIC: Final[list[str]] = [
    "days_until_arrival", "days_since_booking",
    "pct_lead_time_elapsed", "is_within_7d_of_arrival",
]

# --- ratePlan_category map (verbatim port of notebook 00 §3.0.d) ------------
_NONREF_RE: Final = re.compile(
    r"\bnon[_\-\s]*ref|\bnrf\b|\bprepaid\b|nicht.*erstatt|non.*erstatt", re.IGNORECASE)
_PROMO_KEYWORDS: Final[tuple[str, ...]] = (
    "special", "promo", "stay again", "family & friends", "opening", "-%", "%")
_MIN_CATEGORY_COUNT: Final[int] = 50


def _classify_rate_plan(value) -> tuple[str, int, str]:
    """(normalized_name, is_nonref, coarse_type)  verbatim port of nb00 §3.0.d."""
    if value is None or pd.isna(value):
        return "", 0, "unknown"
    n = re.sub(r"\s+", " ", str(value).strip().lower())
    if not n:
        return "", 0, "unknown"
    is_nf = bool(_NONREF_RE.search(n))
    if is_nf:
        t = "nonref"
    elif n == "comp" or n.startswith("comp "):
        t = "comp"
    elif ("corporate" in n or "firmenrate" in n or "consortia" in n or "hrs" in n):
        t = "corporate"
    elif any(k in n for k in _PROMO_KEYWORDS):
        t = "promo"
    elif ("flexible" in n or "midstay" in n or "shortstay" in n
          or "longstay" in n or "airbnb" in n):
        t = "flexible"
    else:
        t = "other"
    return n, int(is_nf), t


def build_rateplan_category_map(windowed: pd.DataFrame, *, name_col: str = "ratePlan_name",
                                arrival_col: str = "arrival",
                                min_count: int = _MIN_CATEGORY_COUNT) -> dict:
    """Fit the `normalized ratePlan_name -> ratePlan_category` map (nb00 §3.0.d).

    `windowed` MUST be the SAME arrival-window population the cleaner filters to
    ([ARRIVAL_FLOOR, cutoff)); the rare-bucket collapse is population-dependent, so
    the caller (data_loader.build_clean_reservations) passes exactly that slice 
    this reproduces the committed map byte-for-byte. Rare buckets (< `min_count` in
    the >28-days-before-max training slice) collapse to 'other'.
    """
    if name_col not in windowed.columns:
        return {}
    raw_s = windowed[name_col].astype("string")
    lookup = {u: _classify_rate_plan(u) for u in raw_s.dropna().unique().tolist()}
    norm_map = {u: v[0] for u, v in lookup.items()}
    nref_map = {u: v[1] for u, v in lookup.items()}
    name = raw_s.map(norm_map).fillna("").astype("string")
    isnr = raw_s.map(nref_map).fillna(0).astype("Int64")
    cat = name.where(name.str.len() > 0, "unknown").mask(isnr == 1, "nonref")
    arr = pd.to_datetime(windowed[arrival_col], utc=True, errors="coerce")
    train_m = arr < (arr.max() - pd.Timedelta(days=28))
    counts = cat[train_m].value_counts()
    protected = {"nonref", "unknown", "other"}
    small = counts[counts < min_count].index.difference(protected)
    cat = cat.where(~cat.isin(small), "other").astype("string")
    return (pd.DataFrame({"ratePlan_name": name, "ratePlan_category": cat}).dropna()
            .drop_duplicates("ratePlan_name")
            .set_index("ratePlan_name")["ratePlan_category"].astype(str).to_dict())


def build_feature_roster(clean: pd.DataFrame, *, rateplan_category_map: dict | None = None) -> dict:
    """Assemble the feature-roster dict from a CLEAN reservations frame  the runtime
    twin of notebook 00 §11, so the roster can be regenerated WITHOUT the notebook.

    Uses the same column-classification helpers the notebook uses
    (`model_feature_roster`, `log_twin_map`, `excluded_columns`) plus the fixed
    `DYNAMIC_NUMERIC` list and the passed-in ratePlan map. Raises on a degenerate
    roster (missing must-have categoricals, dynamic leak, too few numerics)  the
    same guard rails as nb00 §11.
    """
    numeric, categorical = model_feature_roster(clean)
    log_twins = log_twin_map(numeric)
    excluded = excluded_columns()

    leaked = set(DYNAMIC_NUMERIC) & (set(numeric) | set(categorical))
    if leaked:
        raise AssertionError(f"dynamic features leaked into the static roster: {sorted(leaked)}")
    must_cat = {"property_name", "unitGroup_name", "channelCode", "ratePlan_category",
                "guaranteeType", "cancellationFee_name"}
    missing = must_cat - set(categorical)
    if missing:
        raise AssertionError(f"degenerate roster - expected categoricals missing: {sorted(missing)}")
    if len(numeric) < 8:
        raise AssertionError(f"suspiciously few numeric features: {sorted(numeric)}")

    return {
        "generated_by": "src.features.build_feature_roster",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "clean_parquet_rows": int(len(clean)),
        "clean_parquet_cols": int(clean.shape[1]),
        "split": "walk_forward_arrival_anchored_H14 (src.walkforward)",
        "target": "status",
        "numeric": sorted(numeric),
        "categorical": sorted(categorical),
        "log_twins": log_twins,
        "dynamic_numeric": list(DYNAMIC_NUMERIC),
        "excluded": excluded,
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "ratePlan_category_map": dict(rateplan_category_map or {}),
    }


def write_feature_roster(clean: pd.DataFrame, *, rateplan_category_map: dict | None = None,
                         path: str | Path | None = None) -> Path:
    """Build and persist Data/feature_roster.json from a clean frame + rate-plan map.
    Returns the written path. The runtime bootstrap used by build_clean_reservations."""
    p = Path(path) if path is not None else _default_roster_path()
    roster = build_feature_roster(clean, rateplan_category_map=rateplan_category_map)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(roster, fh, indent=2)
    return p
