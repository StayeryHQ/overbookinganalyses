# ---------------------------------------------------------------------------
# Data loader for the Stayery overbooking project.
#
# What this module does:
#   1. Querying BigQuery (`stayery-analytics.reporting.reservations`),
#      excluding PII columns AT THE SQL LEVEL (`SELECT * EXCEPT(...)`) so PII is
#      never pulled into the client at all.
#   2. `strip_pii` as a defense-in-depth safety net (no-op if SQL already excluded).
#   3. Coercing dtypes (BQ returns many numerics/bools as STRING).
#   4. Persisting a PII-free parquet cache so notebooks are fast and free.
#   5. Schema-drift warnings.
#
# Usage:
#     from src import load_reservations, load_clean_reservations
#     df = load_reservations()              # raw (no PII), cached parquet
#     df = load_reservations(force_refresh=True)
#     df = load_reservations(upcoming_only=True)   # for daily scoring
#     df = load_clean_reservations()             # post-cleaning parquet (from 00)
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import pandas as pd

from .paths import data_dir, schema_config_path

logger = logging.getLogger(__name__)

# ---- BigQuery target ------------------------------------------------------

BQ_PROJECT: Final[str] = "stayery-analytics"
BQ_DATASET: Final[str] = "reporting"
BQ_TABLE:   Final[str] = "reservations"

# NOTE: The historical 2022 cut-off has been removed a
MIN_ARRIVAL_DATE: Final[str] = "2000-01-01"

# Two cache files: raw (after PII strip + dtype clean) and post-cleaning
# (produced by 00_data_audit.ipynb). Versioned so we can bust if the PII
# list or dtype rules change.
RAW_CACHE_FILE:   Final[str] = "reservations_raw_no_pii.parquet"
CLEAN_CACHE_FILE: Final[str] = "reservations_clean.parquet"

# ---- Dtype hints ----------------------------------------------------------
# BQ frequently returns these as STRING; we fix them in one place.
INT_COLUMNS: Final[tuple[str, ...]] = (
    "adults", "guest_id", "is_first_res", "is_last_res",
)
DATETIME_COLUMNS: Final[tuple[str, ...]] = (
    "arrival", "departure", "created", "modified",
)
BOOL_COLUMNS: Final[tuple[str, ...]] = (
    "ratePlan_isSubjectToCityTax", "paymentAccount_isVirtual",
    "paymentAccount_isActive",     "allFoliosHaveInvoice",
    "hasCityTax",                  "company_canCheckOutOnAr",
)

# ---- PII columns ---------------------------------------------------------
# These columns identify individual guests or carry sensitive payment data.
# They are EXCLUDED AT THE SQL LEVEL by `_query_bigquery` (SELECT * EXCEPT(...)),
# so they never reach the client; `strip_pii` re-drops them as a safety net.
# Update this list when new PII fields are added upstream.
PII_COLUMNS: Final[tuple[str, ...]] = (
    # Guest names & demographics
    "primaryGuest_title", "primaryGuest_firstName", "primaryGuest_middleInitial",
    "primaryGuest_lastName", "primaryGuest_gender", "primaryGuest_birthDate",
    # Contact details
    "primaryGuest_email", "primaryGuest_phone",
    # Postal address (city is borderline; we drop it — country code is enough
    # geographic signal for cancellation modelling).
    "primaryGuest_address_addressLine1", "primaryGuest_address_postalCode",
    "primaryGuest_address_city",
    # Additional guests
    "additionalGuests_title", "additionalGuests_firstName", "additionalGuests_lastName",
    # Payment account
    "paymentAccount_accountNumber", "paymentAccount_accountHolder",
    "paymentAccount_expiryMonth",   "paymentAccount_expiryYear",
    "paymentAccount_payerEmail",
    # Booker (person who placed the booking)
    "booker_firstName", "booker_lastName", "booker_email", "booker_phone",
    "booker_comment",
    # Free-text fields that routinely contain PII
    "guestComment",
    # Corporate tax IDs
    "primaryGuest_company_taxId",
    # External system identifiers that link back to PII in OTA portals
    "externalCode",
)


# =============================================================================
# Public API
# =============================================================================

def load_reservations(
    force_refresh: bool = False,
    limit: int | None = None,
    upcoming_only: bool = False,
    quiet: bool = False,
) -> pd.DataFrame:
    """Load reservations as a DataFrame, PII-stripped, dtype-cleaned.

    Parameters
    ----------
    force_refresh : bool
        If True, ignore the parquet cache and re-query BigQuery.
    limit : int | None
        Optional row limit; applied at the SQL level for fresh fetches and
        in pandas for cache reads. Useful for fast iteration during development.
    upcoming_only : bool
        If True, only return rows whose `arrival` is in the future. Used by
        the daily scoring notebook. Hits the cache same as the default call —
        it just filters after loading.
    quiet : bool
        Suppress info-level logging.

    Returns
    -------
    pd.DataFrame
        Reservations with PII columns removed and dtypes coerced.
    """
    if not quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache_path = data_dir() / RAW_CACHE_FILE

    if cache_path.exists() and not force_refresh:
        logger.info(f"loading cached parquet: {cache_path.name}")
        df = pd.read_parquet(cache_path)
        if limit is not None:
            df = df.head(limit).copy()
    else:
        if not cache_path.exists():
            logger.info("no parquet cache — querying BigQuery (one-time, slow)")
        else:
            logger.info("force_refresh=True — re-querying BigQuery")
        df = _query_bigquery(limit=limit)
        df = strip_pii(df)
        df = clean_dtypes(df)
        df.to_parquet(cache_path, index=False)
        logger.info(f"cached PII-free parquet → {cache_path.name}")

    if upcoming_only:
        now = pd.Timestamp.utcnow()
        df = df[pd.to_datetime(df["arrival"], utc=True) >= now].copy()

    _validate_schema(df)
    return df


def load_clean_reservations() -> pd.DataFrame:
    """Load the cleaned parquet produced by 00_data_audit.ipynb.

    Raises FileNotFoundError with a helpful message if the file doesn't exist.
    """
    p = data_dir() / CLEAN_CACHE_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run notebooks/00_data_audit.ipynb first to "
            f"produce the cleaned dataset."
        )
    return pd.read_parquet(p)


# =============================================================================
# Helpers (also exported — useful in notebooks for ad-hoc transforms)
# =============================================================================

def strip_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every PII column dropped.

    `errors='ignore'` means we don't crash if a column is already missing —
    silent schema drift in BigQuery shouldn't take the whole pipeline down.
    """
    n_before = df.shape[1]
    out = df.drop(columns=list(PII_COLUMNS), errors="ignore")
    n_dropped = n_before - out.shape[1]
    if n_dropped:
        logger.info(f"PII strip: removed {n_dropped} columns")
    return out


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to sensible dtypes.

    Strategy: be tolerant. Missing columns are skipped; failed conversions
    log a warning and keep the column as-is.
    """
    out = df.copy()

    for col in DATETIME_COLUMNS:
        if col in out.columns:
            try:
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            except Exception as e:  # noqa: BLE001 — defensive
                logger.warning(f"could not parse {col} as datetime: {e}")

    for col in INT_COLUMNS:
        if col in out.columns:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"could not coerce {col} to Int64: {e}")

    truthy = {"true", "True", "TRUE", "1", "yes", "Yes"}
    falsy  = {"false", "False", "FALSE", "0", "no", "No"}
    for col in BOOL_COLUMNS:
        if col not in out.columns:
            continue
        s = out[col]
        if s.dtype == bool:
            out[col] = s.astype("boolean")
            continue
        out[col] = s.map(
            lambda v: True if v in truthy else (False if v in falsy else pd.NA)
        ).astype("boolean")

    return out


# =============================================================================
# Internal
# =============================================================================

def _query_bigquery(limit: int | None = None) -> pd.DataFrame:
    """Lazy-import BigQuery so this module is importable without it installed."""
    try:
        from google.cloud import bigquery  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Run:\n"
            "    uv add google-cloud-bigquery db-dtypes pyarrow\n"
            "and authenticate with `gcloud auth application-default login`."
        ) from e

    client = bigquery.Client()
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    # --- exclude PII AT THE SQL LEVEL (data minimisation) ------------------
    # Never pull PII into the client at all (instead of fetching everything and
    # dropping in pandas). `SELECT * EXCEPT(...)` errors on a column not in the
    # table, so we intersect PII_COLUMNS with the LIVE schema first (robust to
    # upstream schema drift). `strip_pii` downstream stays as a safety net.
    table_cols = {f.name for f in client.get_table(table_ref).schema}
    pii_present = [c for c in PII_COLUMNS if c in table_cols]
    except_clause = f" EXCEPT({', '.join(pii_present)})" if pii_present else ""

    # No date filter - the user wants the full history. Any time-based
    # cuts happen explicitly in 00_data_audit.ipynb.
    sql = f"SELECT *{except_clause} FROM `{table_ref}`"
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    logger.info(
        f"running BigQuery query (full history; {len(pii_present)} PII columns "
        f"excluded at SQL level)…"
    )
    return client.query(sql).to_dataframe()


def _validate_schema(df: pd.DataFrame) -> None:
    """Warn if expected (non-PII) columns are missing vs. the schema JSON."""
    schema_path: Path = schema_config_path()
    if not schema_path.exists():
        logger.warning(f"schema file not found at {schema_path}, skipping validation")
        return
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    expected = {f["name"] for f in schema} - set(PII_COLUMNS)
    actual = set(df.columns)
    missing = expected - actual
    extra   = actual - expected
    if missing:
        logger.warning(
            f"[schema drift] {len(missing)} expected columns MISSING: "
            f"{sorted(missing)[:5]}{'…' if len(missing) > 5 else ''}"
        )
    if extra:
        logger.info(
            f"[schema info] {len(extra)} extra columns present: "
            f"{sorted(extra)[:5]}{'…' if len(extra) > 5 else ''}"
        )
