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
    bigquerystorage.googleapis.com slows the pull down  it no longer breaks it.
    """
    if not _USE_BQ_STORAGE:
        return query_job.to_dataframe(create_bqstorage_client=False)
    try:
        return query_job.to_dataframe()
    except Exception as e:  # noqa: BLE001  Storage endpoint unreachable etc.; fall back to REST
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
# THE key insight (learned the hard way): the JOB project and the DATA project
# are different things. Queries reference fully-qualified tables
# (`stayery-analytics.reporting.…`), so reading only needs dataViewer THERE 
# but the query JOB runs in the client's own project, where the caller needs
# job-creation rights. Pinning the job project to the data project caused
# `403 … serviceusage.services.use` for user accounts. So, like the sibling
# project where this "just works": let ADC decide the job project.
# Mirrors the sibling project 1:1, so the SAME service-account key / ADC file
# behaves identically in both repos (the drive scope is unused here but harmless).
_BQ_SCOPES: Final[tuple[str, ...]] = (
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive.readonly",
)


def get_bigquery_client():
    """Build a BigQuery client from the first available credential source.

    VERBATIM copy of the sibling project's working method:
      1. GCP_SERVICE_ACCOUNT_JSON_FILE   service-account key file (client runs
         in the SA's own project)
      2. GOOGLE_APPLICATION_CREDENTIALS  same handling
      3. gcloud ADC  bare Client(), the SDK resolves project/quota exactly like
         the sibling repo does. NO pinning, NO env juggling on top.

    Job project ≠ data project: queries name the tables fully qualified, so the
    account only needs read access on `stayery-analytics`; jobs run wherever the
    credential's own project is. Use `main.py bqcheck` to see the full resolution.
    """
    from google.cloud import bigquery  # type: ignore[import-untyped]
    from google.oauth2 import service_account  # type: ignore[import-untyped]

    sa_json_file = os.environ.get("GCP_SERVICE_ACCOUNT_JSON_FILE")
    if sa_json_file and Path(sa_json_file).exists():
        creds = service_account.Credentials.from_service_account_file(
            sa_json_file, scopes=list(_BQ_SCOPES)
        )
        return bigquery.Client(credentials=creds, project=creds.project_id)

    sa_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_file and Path(sa_file).exists():
        creds = service_account.Credentials.from_service_account_file(
            sa_file, scopes=list(_BQ_SCOPES)
        )
        return bigquery.Client(credentials=creds, project=creds.project_id)

    # ADC (local gcloud login): enforce nothing  identical to the sibling repo.
    try:
        return bigquery.Client()
    except Exception as e:  # noqa: BLE001  translate into a fixable message
        raise RuntimeError(
            "BigQuery client could not be built from gcloud ADC "
            f"({type(e).__name__}: {e}). Either log in:\n"
            "    gcloud auth application-default login\n"
            "or point GCP_SERVICE_ACCOUNT_JSON_FILE at the service-account key "
            "file the sibling project uses. Run `python main.py bqcheck` for the "
            "full diagnosis."
        ) from e


def bigquery_diagnose() -> list[str]:
    """Every fact that decides WHERE BigQuery jobs run  one line each, best
    effort, never raises. This settles 403 mysteries in one glance: which
    credential source wins, what quota/config project rides along, and which
    project the client would actually create jobs in."""
    lines: list[str] = []
    for env in ("GCP_SERVICE_ACCOUNT_JSON_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        v = os.environ.get(env)
        note = "" if not v else ("  (file exists)" if Path(v).exists() else "  (FILE MISSING!)")
        lines.append(f"{env} = {v or ''}{note}")
    for env in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT"):
        lines.append(f"{env} = {os.environ.get(env) or ''}")
    try:
        import google.auth  # type: ignore[import-untyped]
        creds, adc_project = google.auth.default()
        lines.append(f"ADC credential type  = {type(creds).__name__}")
        lines.append(f"ADC default project  = {adc_project or ''}")
        lines.append(f"ADC quota project    = {getattr(creds, 'quota_project_id', None) or ''}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"ADC = ERROR: {type(e).__name__}: {str(e)[:200]}")
    try:
        import subprocess
        p = subprocess.run(["gcloud", "config", "get-value", "project"],
                           capture_output=True, text=True, timeout=10)
        lines.append(f"gcloud config project = {p.stdout.strip() or ''}")
    except Exception:  # noqa: BLE001
        lines.append("gcloud config project = (gcloud CLI not available)")
    try:
        client = get_bigquery_client()
        lines.append(f"→ JOB project the client will use = {client.project}")
        if client.project == BQ_PROJECT:
            lines.append(
                f"  ⚠ PROBLEM: jobs would run in the DATA project '{BQ_PROJECT}', where "
                "this account may not create jobs (the classic 403). Point the quota/"
                "config project at YOUR OWN project  see hints from bqcheck."
            )
    except Exception as e:  # noqa: BLE001
        lines.append(f"→ client build FAILED: {type(e).__name__}: {str(e)[:200]}")
    return lines


def bigquery_healthcheck() -> dict:
    """Cheap end-to-end probe: build a client, COUNT(*) the reservations table
    (metadata-only, effectively free). NEVER raises  returns {ok, project,
    detail}, on failure with an actionable hint. Used by `main.py bqcheck` and
    the Update page's connection test, so auth problems are visible in seconds
    instead of surfacing halfway through a refresh."""
    try:
        client = get_bigquery_client()
        sql = f"SELECT COUNT(*) AS n FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
        n = int(list(client.query(sql).result(timeout=30))[0][0])
        return {"ok": True, "project": client.project,
                "detail": f"connected (project '{client.project}')  "
                          f"{n:,} rows visible in {BQ_DATASET}.{BQ_TABLE}"}
    except Exception as e:  # noqa: BLE001  a health check must never crash the caller
        msg = str(e)
        low = msg.lower()
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            hint = " → package missing: run `uv sync` (google-cloud-bigquery)."
        elif "serviceusage" in low or "to use project" in low or "quota" in low:
            hint = (" → the query JOB is trying to run in a project where you may not "
                    "create jobs. Jobs do NOT need to run in the data project  point "
                    "them at your own: `gcloud auth application-default set-quota-project "
                    "<your-project>` or `export BQ_BILLING_PROJECT=<your-project>`.")
        elif "default credentials" in low or "no google credentials" in low:
            hint = " → not logged in: run `gcloud auth application-default login`."
        elif "403" in msg or "permission" in low or "access denied" in low:
            hint = (" → the account lacks read permission on the data in "
                    f"'{BQ_PROJECT}' (needs roles/bigquery.dataViewer there).")
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


def load_reservations_upcoming_window(days: int = 14, quiet: bool = False) -> pd.DataFrame:
    """Pull ONLY the next `days` days of arrivals straight from BigQuery (PII stripped,
    dtypes cleaned). Deliberately does NOT read or write the full-history cache  the
    caller (Occupancy scoring) wants fresh upcoming data and only a 14-day SQL scan.
    Fails loudly if BigQuery is unavailable (no cache fallback)."""
    if not quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = _query_bigquery_window(days=days)
    df = strip_pii(df)          # belt-and-suspenders (SQL already EXCEPTs PII)
    df = clean_dtypes(df)
    _validate_schema(df)
    return df


def load_clean_reservations() -> pd.DataFrame:
    """Load the cleaned parquet. Rebuilt programmatically by build_clean_reservations()
    (the CH 'update history' action) or by notebook 00.

    Raises FileNotFoundError with a helpful message if the file doesn't exist.
    """
    p = data_dir() / CLEAN_CACHE_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Rebuild it (Cancellation History → update) or run "
            f"notebooks/00_data_audit.ipynb."
        )
    return pd.read_parquet(p)


# --- Clean/label pipeline (programmatic equivalent of notebook 00) -----------
# Mirrors 00 §2.6/§3.0/§4.x/§7. Feature engineering REUSES src.scoring.build_features 
# the SAME function used at serving  so training and serving cannot drift.
ARRIVAL_FLOOR: Final[pd.Timestamp] = pd.Timestamp("2022-08-01", tz="UTC")
GRACE_DAYS: Final[int] = 2                     # today - GRACE_DAYS = arrival cutoff
RESOLVED_STATUSES: Final[frozenset] = frozenset({"Canceled", "CheckedOut", "InHouse", "NoShow"})

# Final clean schema (order + dtype)  matches Data/reservations_clean.parquet.
_CLEAN_DTYPES: Final[dict] = {
    "status": "int8", "property_name": "object", "ratePlan_isSubjectToCityTax": "boolean",
    "unitGroup_name": "object", "channelCode": "string", "guaranteeType": "object",
    "cancellationFee_name": "object", "arrival": "datetime64[us, UTC]",
    "created": "datetime64[us, UTC]", "lead_time_days": "float64", "los_nights": "Int64",
    "gross_amount": "float64", "gross_per_night": "Float64", "ratePlan_category": "string",
    "has_group": "Int64", "diff_gross_cancellation_fee": "float64", "adults_n": "Int64",
    "has_promo": "Int64", "has_corporate_code": "Int64", "has_children": "Int64",
    "arrival_dow": "Int64", "arrival_month": "Int64", "is_weekend_arrival": "Int64",
    "stay_bucket": "string", "log_gross_amount": "float64", "los_nights_log": "Float64",
    "lead_time_days_log": "float64", "gross_per_night_log": "Float64",
    "diff_gross_cancellation_fee_log": "float64", "cancel_days_before_arrival": "float64",
    "is_canceled_by_arrival": "int8", "outcome_known_date": "datetime64[ns, UTC]",
}


def build_clean_reservations(raw: pd.DataFrame, *, write_roster: bool = False) -> pd.DataFrame:
    """Raw PII-free reservations -> the cleaned/labelled training frame (same schema as
    Data/reservations_clean.parquet). Programmatic equivalent of notebook 00's cleaning.

    Validated to reproduce the clean cache's feature columns + target exactly. Feature
    engineering reuses src.scoring.build_features (the serving function), so there is ONE
    definition  training and serving cannot drift.

    The ratePlan_category map is fit here (on the SAME arrival-window population the
    notebook uses) and handed to build_features explicitly  so this runs WITHOUT a
    pre-existing feature_roster.json. If `write_roster` is True (or no roster exists yet),
    it also (re)writes Data/feature_roster.json, so a fresh deploy can self-heal instead
    of hard-failing on a missing roster (was: notebook-only artifact).
    """
    from .scoring import build_features
    from .features import (ROSTER_FILENAME, build_rateplan_category_map,
                           write_feature_roster)
    from . import walkforward as wf

    df = raw.copy()
    arr = pd.to_datetime(df["arrival"], utc=True, errors="coerce")
    dep = pd.to_datetime(df["departure"], utc=True, errors="coerce")
    cre0 = pd.to_datetime(df["created"], utc=True, errors="coerce")
    ct = pd.to_datetime(df.get("cancellationTime"), utc=True, errors="coerce")

    # §2.6 arrival window  RUN_TIMESTAMP anchored on the data (reproducible), not wall-clock.
    cutoff = cre0.max().normalize() - pd.Timedelta(days=GRACE_DAYS)
    # §4.2 negative-lead clip: created := arrival (booking can't be made after arrival).
    neg = (cre0 > arr).fillna(False)
    df.loc[neg, "created"] = arr[neg]
    cre = pd.to_datetime(df["created"], utc=True, errors="coerce")

    df["cancel_days_before_arrival"] = (arr - ct) / pd.Timedelta(days=1)
    lead = (arr - cre) / pd.Timedelta(days=1)
    los = (dep.dt.normalize() - arr.dt.normalize()) / pd.Timedelta(days=1)
    gross = pd.to_numeric(df.get("totalGrossAmount_amount"), errors="coerce")

    # §4.3 row rules + resolved-status filter (Confirmed dropped: label not yet known).
    keep = (
        (arr >= ARRIVAL_FLOOR) & (arr < cutoff)
        & (lead >= 0) & (lead <= 365)
        & (los >= 1) & (los <= 200)
        & (~gross.le(0).fillna(True))
        & (df["status"].isin(RESOLVED_STATUSES))
    )
    d = df[keep].copy()

    # ratePlan_category map  fit on the SAME arrival-window population the notebook uses
    # ([ARRIVAL_FLOOR, cutoff)), so the rare-bucket collapse reproduces the committed map.
    window_mask = (arr >= ARRIVAL_FLOOR) & (arr < cutoff)
    rp_map = build_rateplan_category_map(df[window_mask])

    # §4.5 target: 1 = Canceled AND logged at/before arrival (same-day counts as positive).
    cdba = pd.to_numeric(d["cancel_days_before_arrival"], errors="coerce")
    is_cba = (d["status"].eq("Canceled") & cdba.ge(0)).astype("int8")

    feat = build_features(d, rateplan_category_map=rp_map)   # §3.0 features  shared with serving
    feat = wf.add_outcome_known_date(feat)    # §7 split metadata
    feat["cancel_days_before_arrival"] = d["cancel_days_before_arrival"].to_numpy()
    feat["is_canceled_by_arrival"] = is_cba.to_numpy()
    feat["status"] = is_cba.to_numpy()        # encode the target in-place (was the string status)

    out = feat[list(_CLEAN_DTYPES)].copy()
    for col, dt in _CLEAN_DTYPES.items():
        try:
            out[col] = out[col].astype(dt)
        except (TypeError, ValueError):
            pass                              # tolerate minor dtype drift; values are what matter
    out = out.reset_index(drop=True)

    # Self-heal the roster: write it when asked, or when it is simply missing (fresh
    # deploy). The map came from the arrival-window population above, so the persisted
    # roster matches what the notebook would have produced.
    if write_roster or not (data_dir() / ROSTER_FILENAME).exists():
        write_feature_roster(out, rateplan_category_map=rp_map)
    return out


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
    """(sql, n_pii) for THE full-history reservations pull  PII excluded in SQL.
    Used for the history views + retraining. Scoring the upcoming window uses the
    windowed variant below (_reservations_window_query).
    """
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    table_cols = {f.name for f in client.get_table(table_ref).schema}
    pii_present = [c for c in PII_COLUMNS if c in table_cols]
    except_clause = f" EXCEPT({', '.join(pii_present)})" if pii_present else ""

    sql = f"SELECT *{except_clause} FROM `{table_ref}`"
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return sql, len(pii_present)


def _reservations_window_query(client, *, days: int = 14):
    """(sql, n_pii) for a forward-WINDOWED reservations pull: only rows whose
    `arrival` falls in [now, now + `days`). PII excluded in SQL. This is what the
    Occupancy scoring action uses so it hits ONLY the next `days` days in BigQuery,
    never the full history.

    The `arrival` column's BigQuery type is detected at build time so the WHERE clause
    uses the matching CURRENT_*/*_ADD functions (DATE / DATETIME / TIMESTAMP), avoiding
    a type-mismatch error and keeping partition pruning intact.
    """
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    schema = client.get_table(table_ref).schema
    table_cols = {f.name for f in schema}
    pii_present = [c for c in PII_COLUMNS if c in table_cols]
    except_clause = f" EXCEPT({', '.join(pii_present)})" if pii_present else ""

    arr_type = next((f.field_type for f in schema if f.name == "arrival"), "TIMESTAMP").upper()
    now_fn, add_fn = {
        "DATE": ("CURRENT_DATE()", "DATE_ADD"),
        "DATETIME": ("CURRENT_DATETIME()", "DATETIME_ADD"),
    }.get(arr_type, ("CURRENT_TIMESTAMP()", "TIMESTAMP_ADD"))

    where = (f"WHERE arrival >= {now_fn} "
             f"AND arrival < {add_fn}({now_fn}, INTERVAL {int(days)} DAY)")
    sql = f"SELECT *{except_clause} FROM `{table_ref}`\n{where}"
    return sql, len(pii_present)


def _query_bigquery_window(days: int = 14) -> pd.DataFrame:
    """Run the forward-windowed reservations query (next `days` days of arrivals).
    Fails LOUDLY: no cache fallback  the caller explicitly wants FRESH upcoming data.
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
    sql, n_pii = _reservations_window_query(client, days=days)
    logger.info(
        f"running WINDOWED BigQuery query (next {days} days of arrivals; "
        f"{n_pii} PII columns excluded at SQL level)…"
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
            f"BigQuery windowed query timed out or failed after {timeout_seconds}s: {e}"
        ) from e

    return _download_df(query_job)


def _query_bigquery(limit: int | None = None) -> pd.DataFrame:
    """Run the full-history reservations query. Fails LOUDLY: no cache fallback
    here  callers that want the cache read it explicitly via load_reservations.
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
# Property performance (occupancy / daily operations)  second BigQuery table
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

# Allow-list  the ONLY columns we select from the table (operational + ADR).
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
# schema above. NOTE: netAdr_amount is intentionally NOT here  see above.
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
    """Location universe from the performance table  replaces configs/locations.yaml.

    Returns one row per propertyId with `units` = the most recent houseCount
    (the property's bookable unit count). New properties appear automatically.
    Returns an EMPTY frame (columns propertyId/units) if the table/cache is
    unavailable, so callers can fall back to the YAML/dummy.
    """
    try:
        perf = load_property_performance(force_refresh=force_refresh, quiet=True)
    except Exception as e:  # noqa: BLE001  no creds / no table / offline
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
    """Mean ADR (`netAdr_amount`) per propertyId  used to PRE-FILL the empty-room
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
