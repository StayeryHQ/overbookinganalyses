# ---------------------------------------------------------------------------
# Data loader for the Stayery overbooking project.
#
#   1. Querying BigQuery (`stayery-analytics.reporting.reservations` and performance analytics),
#      excluding PII columns at SQL level (`SELECT * EXCEPT(...)`)
#   2. `strip_pii` as a safety net
#   3. Coercing dtypes
#   4. Persisting a PII-free parquet cache so notebooks are fast
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
import os
from pathlib import Path
from typing import Final

import pandas as pd

from .paths import data_dir, schema_config_path

logger = logging.getLogger(__name__)

# The BigQuery *Storage* API (host bigquerystorage.googleapis.com) is a SEPARATE endpoint the
# client can use to download results faster. It is OPT-IN here (default: plain SDK download
# over bigquery.googleapis.com via `to_dataframe(create_bqstorage_client=False)`), because the
# Storage endpoint is blocked in some environments and the failed round-trip costs time and
# produces confusing errors. Set BQ_USE_STORAGE_API=1 to enable the faster Storage download
# when you know the host is reachable; on ANY failure it still degrades to the plain download.
_USE_BQ_STORAGE: Final[bool] = os.environ.get(
    "BQ_USE_STORAGE_API", "0"
).strip().lower() in ("1", "true", "yes", "on")
# Background refreshes should not hang forever if BigQuery is slow or blocked. The page uses
# this timeout to fail fast and fall back to the local cache instead of leaving the UI in a
# permanent loading state.
DEFAULT_BQ_QUERY_TIMEOUT_SECONDS: Final[int] = 90


def _get_bq_query_timeout_seconds() -> int:
    """Read a runtime override for the BigQuery query timeout, defaulting to 90s."""
    raw = os.environ.get(
        "BQ_QUERY_TIMEOUT_SECONDS", str(DEFAULT_BQ_QUERY_TIMEOUT_SECONDS)
    ).strip()
    try:
        return int(raw) if raw else DEFAULT_BQ_QUERY_TIMEOUT_SECONDS
    except ValueError:
        logger.warning(
            "invalid BQ_QUERY_TIMEOUT_SECONDS=%r; using default %ds",
            raw,
            DEFAULT_BQ_QUERY_TIMEOUT_SECONDS,
        )
        return DEFAULT_BQ_QUERY_TIMEOUT_SECONDS


def _download_df(query_job) -> pd.DataFrame:
    """Materialise a BigQuery job as a DataFrame, resilient to a blocked Storage API endpoint.

    Prefers the fast BigQuery Storage API; on ANY failure (or when BQ_USE_STORAGE_API=0) it
    retries the plain REST download (`create_bqstorage_client=False`). So a missing route to
    bigquerystorage.googleapis.com slows the pull down — it no longer breaks it.
    """
    if not _USE_BQ_STORAGE:
        return query_job.to_dataframe(create_bqstorage_client=False)
    try:
        return query_job.to_dataframe()
    except Exception as e:  # noqa: BLE001 — Storage endpoint unreachable etc.; fall back to REST
        logger.warning(
            f"BigQuery Storage API download failed ({e}); retrying via REST "
            "(create_bqstorage_client=False). Set BQ_USE_STORAGE_API=0 to skip this attempt."
        )
        return query_job.to_dataframe(create_bqstorage_client=False)


# ---- BigQuery target ------------------------------------------------------

BQ_PROJECT: Final[str] = "stayery-analytics"
BQ_DATASET: Final[str] = "reporting"
BQ_TABLE: Final[str] = "reservations"

# ---- BigQuery client (the ONE construction point) ---------------------------
# Scopes are requested only for SERVICE-ACCOUNT key files. SAs are not subject to
# the user-consent block, so the extra Drive scope lets them read Drive/Sheet-backed
# external tables too (none in this project today — harmless and future-proof).
# The local gcloud ADC path enforces NO scopes; it uses whatever was granted at login.
_BQ_SCOPES: Final[tuple[str, ...]] = (
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive.readonly",
)


def get_bigquery_client():
    """Build a BigQuery client from the first available credential source.

    Plain google-cloud-bigquery SDK — no custom REST calls, no Storage API needed.
    Resolution order:
      1. GCP_SERVICE_ACCOUNT_JSON_FILE  — explicit service-account key file
      2. GOOGLE_APPLICATION_CREDENTIALS — the standard Google env var, same handling
      3. gcloud ADC (`gcloud auth application-default login`)

    The JOB (billing) project is always pinned to BQ_PROJECT. ADC user credentials
    additionally carry a QUOTA project: whatever `gcloud auth application-default
    set-quota-project <p>` stored, overridable via GOOGLE_CLOUD_QUOTA_PROJECT.
    Missing credentials raise a RuntimeError with the exact commands to fix —
    never a silent misconfiguration.
    """
    from google.cloud import bigquery  # type: ignore[import-untyped]
    from google.oauth2 import service_account  # type: ignore[import-untyped]

    for env in ("GCP_SERVICE_ACCOUNT_JSON_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        key_file = os.environ.get(env)
        if key_file and Path(key_file).exists():
            creds = service_account.Credentials.from_service_account_file(
                key_file, scopes=list(_BQ_SCOPES)
            )
            return bigquery.Client(
                credentials=creds, project=creds.project_id or BQ_PROJECT
            )

    # ADC (local gcloud login). Load explicitly so a missing login fails with a
    # fixable message instead of a generic stack trace deep inside the client.
    import google.auth  # type: ignore[import-untyped]
    import google.auth.exceptions  # type: ignore[import-untyped]

    try:
        creds, _ = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise RuntimeError(
            "No Google credentials found. Either point GCP_SERVICE_ACCOUNT_JSON_FILE "
            "at a service-account key file, or log in once:\n"
            "    gcloud auth application-default login\n"
            f"    gcloud auth application-default set-quota-project {BQ_PROJECT}"
        ) from e
    quota = os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT")
    if quota and hasattr(creds, "with_quota_project"):
        creds = creds.with_quota_project(quota)
    return bigquery.Client(credentials=creds, project=BQ_PROJECT)


def bigquery_healthcheck() -> dict:
    """Cheap end-to-end probe: build a client, COUNT(*) the reservations table
    (metadata-only, effectively free). NEVER raises — returns {ok, project,
    detail}, on failure with an actionable hint. Used by `main.py bqcheck` and
    the Update page's connection test, so auth problems are visible in seconds
    instead of surfacing halfway through a refresh."""
    try:
        client = get_bigquery_client()
        sql = f"SELECT COUNT(*) AS n FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
        n = int(list(client.query(sql).result(timeout=30))[0][0])
        return {"ok": True, "project": client.project,
                "detail": f"connected (project '{client.project}') — "
                          f"{n:,} rows visible in {BQ_DATASET}.{BQ_TABLE}"}
    except Exception as e:  # noqa: BLE001 — a health check must never crash the caller
        msg = str(e)
        low = msg.lower()
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            hint = " → package missing: run `uv sync` (google-cloud-bigquery)."
        elif "quota" in low or "userproject" in low or "user project" in low:
            hint = (" → quota-project problem: run `gcloud auth application-default "
                    f"set-quota-project {BQ_PROJECT}` or export GOOGLE_CLOUD_QUOTA_PROJECT.")
        elif "default credentials" in low or "no google credentials" in low:
            hint = " → not logged in: run `gcloud auth application-default login`."
        elif "403" in msg or "permission" in low or "access denied" in low:
            hint = (" → the account lacks BigQuery read permission on "
                    f"'{BQ_PROJECT}' (needs roles/bigquery.jobUser + dataViewer).")
        elif "timed out" in low or "timeout" in low or "deadline" in low:
            hint = " → network problem: check VPN/firewall towards bigquery.googleapis.com."
        else:
            hint = ""
        return {"ok": False, "project": None,
                "detail": f"{type(e).__name__}: {msg[:400]}{hint}"}


# Two cache files: raw (after PII strip + dtype clean) and post-cleaning
# (produced by 00_data_audit.ipynb).
RAW_CACHE_FILE: Final[str] = "reservations_raw_no_pii.parquet"
CLEAN_CACHE_FILE: Final[str] = "reservations_clean.parquet"

# ---- Dtype hints ----------------------------------------------------------
INT_COLUMNS: Final[tuple[str, ...]] = (
    "adults",
    "guest_id",
    "is_first_res",
    "is_last_res",
)
DATETIME_COLUMNS: Final[tuple[str, ...]] = (
    "arrival",
    "departure",
    "created",
    "modified",
)
BOOL_COLUMNS: Final[tuple[str, ...]] = (
    "ratePlan_isSubjectToCityTax",
    "paymentAccount_isVirtual",
    "paymentAccount_isActive",
    "allFoliosHaveInvoice",
    "hasCityTax",
    "company_canCheckOutOnAr",
)

# ---- PII columns ---------------------------------------------------------
# These columns identify individual guests or carry sensitive payment data.
# They are excluded at SQL level
PII_COLUMNS: Final[tuple[str, ...]] = (
    "primaryGuest_title",
    "primaryGuest_firstName",
    "primaryGuest_middleInitial",
    "primaryGuest_lastName",
    "primaryGuest_gender",
    "primaryGuest_birthDate",
    "primaryGuest_email",
    "primaryGuest_phone",
    "primaryGuest_address_addressLine1",
    "primaryGuest_address_postalCode",
    "primaryGuest_address_city",
    "additionalGuests_title",
    "additionalGuests_firstName",
    "additionalGuests_lastName",
    "paymentAccount_accountNumber",
    "paymentAccount_accountHolder",
    "paymentAccount_expiryMonth",
    "paymentAccount_expiryYear",
    "paymentAccount_payerEmail",
    "booker_firstName",
    "booker_lastName",
    "booker_email",
    "booker_phone",
    "booker_comment",
    "guestComment",
    "primaryGuest_company_taxId",
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
        the daily scoring notebook. Hits the cache same as the default call -
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
            logger.info("no parquet cache - querying BigQuery")
        else:
            logger.info("force_refresh=True re-querying BigQuery")
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
# Helpers (also exported - useful in notebooks for ad-hoc transforms)
# =============================================================================


def strip_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every PII column dropped.

    `errors='ignore'` means we don't crash if a column is already missing
    silent schema drift in BigQuery shouldn't take the whole pipeline down.
    """
    n_before = df.shape[1]
    out = df.drop(columns=list(PII_COLUMNS), errors="ignore")
    n_dropped = n_before - out.shape[1]
    if n_dropped:
        logger.info(
            f"PII strip: removed {n_dropped} columns using function - review SQL query"
        )
    return out


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to sensible dtypes.

    Strategy: Missing columns are skipped and failed conversions
    log a warning and keep the column as-is.
    """
    out = df.copy()

    for col in DATETIME_COLUMNS:
        if col in out.columns:
            try:
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            except Exception as e:
                logger.warning(f"could not parse {col} as datetime: {e}")

    for col in INT_COLUMNS:
        if col in out.columns:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"could not coerce {col} to Int64: {e}")

    truthy = {"true", "True", "TRUE", "1", "yes", "Yes"}
    falsy = {"false", "False", "FALSE", "0", "no", "No"}
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


def _reservations_query(client, *, limit: int | None = None):
    """(sql, n_pii) for THE one reservations pull — full history, PII excluded
    in SQL. There is deliberately no date-windowed variant anymore: the full
    pull is cheap (~200k rows) and already contains the upcoming bookings, so
    one query serves history AND scoring (no second query definition to drift).
    """
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    table_cols = {f.name for f in client.get_table(table_ref).schema}
    pii_present = [c for c in PII_COLUMNS if c in table_cols]
    except_clause = f" EXCEPT({', '.join(pii_present)})" if pii_present else ""

    sql = f"SELECT *{except_clause} FROM `{table_ref}`"
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return sql, len(pii_present)


def _query_bigquery(limit: int | None = None) -> pd.DataFrame:
    """Run the full-history reservations query. Fails LOUDLY: no cache fallback
    here — callers that want the cache read it explicitly via load_reservations.
    BigQuery imports lazily so the module works without the package installed."""
    try:
        from google.cloud import bigquery  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Run:\n"
            "    uv add google-cloud-bigquery db-dtypes pyarrow\n"
            "and authenticate with `gcloud auth application-default login`."
        ) from e

    client = get_bigquery_client()
    sql, n_pii = _reservations_query(client, limit=limit)
    logger.info(
        f"running BigQuery query (full history; {n_pii} PII columns excluded at SQL level)…"
    )

    timeout_seconds = _get_bq_query_timeout_seconds()
    job = None
    try:
        job = client.query(sql)
        query_job = job.result(timeout=timeout_seconds)
    except Exception as e:  # noqa: BLE001
        if job is not None:
            try:
                job.cancel()
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            f"BigQuery query timed out or failed after {timeout_seconds}s: {e}"
        ) from e

    return _download_df(query_job)


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
    extra = actual - expected
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


# =============================================================================
# Property performance (occupancy / daily operations) — second BigQuery table
# =============================================================================
# `reporting.property_performance_daily` holds one row per property per business
# day. Live schema (confirmed 2026-07-02):
#   businessDay, houseCount, soldCount, outOfOrderCount, arrivalsCount,
#   departuresCount, noShowsCount, cancellationsCount, occupancyPercentage,
#   netUnitRevenue_amount, netAccommodationRevenue_amount,
#   netFoodAndBeveragesRevenue_amount, netOtherRevenue_amount, netAdr_amount,
#   revPar_amount, propertyId
# We pull the operational columns (occupancy + counts) PLUS one documented
# exception: `netAdr_amount` (average daily rate). ADR is technically a revenue
# figure, but the Occupancy dashboard uses it to PRE-FILL the "cost of an empty
# room" parameter, so it earns its place on the allow-list. Every OTHER revenue
# column is still never listed -> never scanned/pulled (data minimisation + cost).
PROPERTY_PERF_TABLE: Final[str] = "property_performance_daily"
PROPERTY_PERF_CACHE: Final[str] = "property_performance_daily.parquet"

# Allow-list — the ONLY columns we select from the table (operational + ADR).
PROP_PERF_COLUMNS: Final[tuple[str, ...]] = (
    "businessDay",
    "houseCount",
    "soldCount",
    "outOfOrderCount",
    "arrivalsCount",
    "departuresCount",
    "noShowsCount",
    "cancellationsCount",
    "occupancyPercentage",
    "netAdr_amount",
    "propertyId",
)
# Remaining revenue columns present in the table but DELIBERATELY excluded
# (documentation only; never referenced by the query). Keep in sync with the live
# schema above. NOTE: netAdr_amount is intentionally NOT here — see above.
PROP_PERF_REVENUE_EXCLUDED: Final[tuple[str, ...]] = (
    "netUnitRevenue_amount",
    "netAccommodationRevenue_amount",
    "netFoodAndBeveragesRevenue_amount",
    "netOtherRevenue_amount",
    "revPar_amount",
)
# Integer-count columns (nullable Int64 so a missing value never becomes 0/NaN-float).
_PERF_INT_COLS: Final[tuple[str, ...]] = (
    "houseCount",
    "soldCount",
    "outOfOrderCount",
    "arrivalsCount",
    "departuresCount",
    "noShowsCount",
    "cancellationsCount",
)


def load_property_performance(
    force_refresh: bool = False, quiet: bool = False
) -> pd.DataFrame:
    """Load `reporting.property_performance_daily` (occupancy + ops), dtype-cleaned.

    Only the operational columns are selected (revenue ignored). Cached to parquet
    like `load_reservations`. `force_refresh=True` re-queries BigQuery. dtypes:
    businessDay -> datetime64[UTC], counts -> Int64, occupancyPercentage -> float64,
    propertyId -> string. Robust to schema drift (missing columns are skipped).
    """
    if not quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache_path = data_dir() / PROPERTY_PERF_CACHE

    if cache_path.exists() and not force_refresh:
        logger.info(f"loading cached parquet: {cache_path.name}")
        df = pd.read_parquet(cache_path)
    else:
        logger.info("querying BigQuery: property_performance_daily (ops columns only)…")
        df = _query_property_performance()
        df = _clean_perf_dtypes(df)
        df.to_parquet(cache_path, index=False)
        logger.info(f"cached property performance → {cache_path.name}")
    return df


def _clean_perf_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the property-performance columns to robust dtypes (skip missing)."""
    out = df.copy()
    if "businessDay" in out.columns:  # business day -> UTC datetime
        out["businessDay"] = pd.to_datetime(
            out["businessDay"], utc=True, errors="coerce"
        )
    for c in _PERF_INT_COLS:  # counts -> nullable Int64
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("occupancyPercentage", "netAdr_amount"):  # occupancy % + ADR -> float
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "propertyId" in out.columns:  # property code -> string
        out["propertyId"] = out["propertyId"].astype("string")
    return out


def _query_property_performance(limit: int | None = None) -> pd.DataFrame:
    """Query BigQuery for the property-performance allow-list columns only."""
    try:
        import google.cloud.bigquery  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Run `uv add google-cloud-bigquery`."
        ) from e
    client = get_bigquery_client()
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{PROPERTY_PERF_TABLE}"
    # Only-needed columns; intersect with the live schema so a renamed/missing
    # column never breaks the query (robust to upstream drift). Revenue columns
    # are simply never listed -> never pulled.
    table_cols = {f.name for f in client.get_table(table_ref).schema}
    cols = [c for c in PROP_PERF_COLUMNS if c in table_cols]
    if not cols:
        raise RuntimeError(f"none of {PROP_PERF_COLUMNS} found in {table_ref}")
    sql = f"SELECT {', '.join(cols)} FROM `{table_ref}`"
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    logger.info(f"  selecting {len(cols)} ops columns from {PROPERTY_PERF_TABLE}…")

    timeout_seconds = _get_bq_query_timeout_seconds()
    job = None
    try:
        job = client.query(sql)
        query_job = job.result(timeout=timeout_seconds)
    except Exception as e:  # noqa: BLE001
        if job is not None:
            try:
                job.cancel()
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            f"BigQuery property-performance query timed out or failed after {timeout_seconds}s: {e}"
        ) from e

    return _download_df(query_job)


def property_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Location universe from the performance table — replaces configs/locations.yaml.

    Returns one row per propertyId with `units` = the most recent houseCount
    (the property's bookable unit count). New properties appear automatically.
    Returns an EMPTY frame (columns propertyId/units) if the table/cache is
    unavailable, so callers can fall back to the YAML/dummy.
    """
    try:
        perf = load_property_performance(force_refresh=force_refresh, quiet=True)
    except Exception as e:  # noqa: BLE001 — no creds / no table / offline
        logger.warning(
            f"property_universe: performance table unavailable ({e}); empty universe"
        )
        return pd.DataFrame(columns=["propertyId", "units"])
    if perf.empty or "propertyId" not in perf.columns:
        return pd.DataFrame(columns=["propertyId", "units"])
    # Latest businessDay row per property gives the current houseCount = units.
    perf = perf.sort_values("businessDay")
    latest = perf.groupby("propertyId", as_index=False).last()
    out = latest[["propertyId"]].copy()
    out["units"] = (
        pd.to_numeric(latest.get("houseCount"), errors="coerce")
        .fillna(0)
        .astype(int)
        .values
    )
    return out.reset_index(drop=True)


def average_room_rate_by_property(
    force_refresh: bool = False, lookback_days: int | None = 90
) -> dict[str, float]:
    """Mean ADR (`netAdr_amount`) per propertyId — used to PRE-FILL the empty-room
    cost in the Occupancy dashboard.

    `lookback_days` restricts the average to the most recent N business days per
    property (None = whole history). Returns {propertyId: adr}; an EMPTY dict if the
    performance cache / BigQuery is unavailable or ADR is absent, so callers pre-fill
    nothing rather than guessing.

    NOTE: keyed by `propertyId` (the performance table's code). Map to
    property_name via the reservations cache's `property_code` column
    (see dash_app.backend.data_access._property_code_to_name).
    """
    try:
        perf = load_property_performance(force_refresh=force_refresh, quiet=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"average_room_rate_by_property: performance table unavailable ({e})"
        )
        return {}
    if (
        perf.empty
        or "netAdr_amount" not in perf.columns
        or "propertyId" not in perf.columns
    ):
        return {}
    df = perf.dropna(subset=["netAdr_amount"]).copy()
    if lookback_days is not None and "businessDay" in df.columns and not df.empty:
        cutoff = pd.to_datetime(df["businessDay"], utc=True).max() - pd.Timedelta(
            days=lookback_days
        )
        df = df[pd.to_datetime(df["businessDay"], utc=True) >= cutoff]
    if df.empty:
        return {}
    means = df.groupby("propertyId")["netAdr_amount"].mean()
    return {str(k): round(float(v), 2) for k, v in means.items() if pd.notna(v)}
