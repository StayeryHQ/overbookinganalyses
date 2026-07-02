# dash_app/backend/data_access.py
# Read-only accessors the Occupancy dashboard uses. EVERYTHING here reads from the
# Phase-1 local caches (parquet) — no live BigQuery is ever triggered by a filter or
# table interaction (hard performance requirement). The only write path is
# refresh_scored(), which re-runs the model on the already-cached reservations and
# is meant to be called from a background callback, never inline on a filter.

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

import src
from src import scoring as sc

# ---- Constants -------------------------------------------------------------
WINDOW_DAYS = 14                      # fixed forward-looking window for this page
DEFAULT_RISK_THRESHOLD = 0.50         # KPI "high-risk bookings" default cut (UI-editable)
RAW_CACHE_FILE = "reservations_raw_no_pii.parquet"
SCORED_CACHE_FILE = "scored_upcoming.parquet"
CLEAN_META_FILE = "reservations_clean_meta.json"

# Statuses that OCCUPY a room on a given night (for the room-type occupancy view).
OCCUPYING_STATUSES = ("Confirmed", "InHouse")


def _data_dir() -> Path:
    return src.data_dir()


# ---- Time window -----------------------------------------------------------
def today_utc() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").normalize()


def window_bounds(today: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) covering the next WINDOW_DAYS days from `today` (UTC midnight)."""
    t = today or today_utc()
    return t, t + pd.Timedelta(days=WINDOW_DAYS)


# ---- Freshness / metadata (for the KPI tiles) ------------------------------
def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def data_freshness() -> dict:
    """When the local caches were last written (file mtimes). None if never built."""
    return {
        "reservations": _mtime(_data_dir() / RAW_CACHE_FILE),
        "scored": _mtime(_data_dir() / SCORED_CACHE_FILE),
    }


def model_meta() -> dict:
    """Real model metadata for the KPI tiles — never fabricated.

    Returns retrained_at + training-set size for the DEFAULT scoring model, reading
    the model card and the cleaned-dataset metadata. Any value that isn't present in
    a real artifact is returned as None so the UI can say "unavailable" explicitly.
    """
    out = {"model": None, "retrained_at": None, "trained_on_bookings": None,
           "trained_on_note": None}
    try:
        name = sc.resolve_model()
    except Exception:  # noqa: BLE001 — no model artifact on disk
        return out
    out["model"] = name
    # retrained_at from the model card, if the card exists and has it.
    try:
        card_path = src.repo_root() / sc.MODEL_REGISTRY[name]["card"]
        if card_path.exists():
            card = json.loads(card_path.read_text())
            out["retrained_at"] = card.get("retrained_at")
    except Exception:  # noqa: BLE001
        pass
    # Training-set size: the hazard card stores person-periods, not bookings, so we
    # report the cleaned-dataset booking count it was trained on (real metadata) and
    # label it as such.
    try:
        meta_path = _data_dir() / CLEAN_META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            # reservations_clean_meta.json uses "rows"; be tolerant of other keys.
            n = meta.get("rows") or meta.get("clean_parquet_rows") or meta.get("n_rows")
            if n is not None:
                out["trained_on_bookings"] = int(n)
                out["trained_on_note"] = "bookings in the cleaned training set"
    except Exception:  # noqa: BLE001
        pass
    return out


# ---- Reservations cache ----------------------------------------------------
@lru_cache(maxsize=1)
def _reservations_cached() -> pd.DataFrame:
    p = _data_dir() / RAW_CACHE_FILE
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def property_list() -> list[str]:
    """Distinct property names from the reservations cache (sorted). [] if no cache."""
    df = _reservations_cached()
    if df.empty or "property_name" not in df.columns:
        return []
    return sorted(df["property_name"].dropna().unique().tolist())


# ---- Scored set (model output) --------------------------------------------
def scored_cache_exists() -> bool:
    return (_data_dir() / SCORED_CACHE_FILE).exists()


def load_scored() -> pd.DataFrame:
    """Read the cached scored upcoming bookings. Empty frame if not scored yet."""
    p = _data_dir() / SCORED_CACHE_FILE
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def refresh_scored(model_name: str | None = None) -> int:
    """Re-run the model over the cached reservations and rewrite the scored parquet.

    Reads reservations from the local cache (force_refresh=False => NO BigQuery).
    Returns the row count. Intended to be called from a BACKGROUND callback (model
    inference can take >1s), never inline on a filter interaction.
    """
    scored = sc.score_upcoming(model_name=model_name, force_refresh=False, save=True)
    _reservations_cached.cache_clear()
    return int(len(scored))


# ---- Window filtering ------------------------------------------------------
def in_window(df: pd.DataFrame, properties: list[str] | None = None,
              today: pd.Timestamp | None = None,
              arrival_col: str = "arrival") -> pd.DataFrame:
    """Filter a frame to arrivals within the 14-day window and the selected
    properties. Safe on an empty frame."""
    if df.empty or arrival_col not in df.columns:
        return df
    start, end = window_bounds(today)
    arr = pd.to_datetime(df[arrival_col], utc=True)
    mask = (arr >= start) & (arr < end)
    if properties:
        mask &= df["property_name"].isin(properties)
    return df.loc[mask].copy()


# ---- Per-arrival-night expected freed rooms (for the overbooking rec) ------
def per_night_expected_freed(scored_window: pd.DataFrame,
                             hotel_col: str | None = "property_name") -> pd.DataFrame:
    """Aggregate per-booking cancel probabilities to per-(arrival-night[, hotel])
    EXPECTED freed rooms. Mirrors src.hazard.per_night_table (inlined to avoid the
    heavier import): exp = Σp, var = Σp(1-p) (Poisson-binomial), n = bookings.
    """
    if scored_window.empty or "cancel_proba" not in scored_window.columns:
        cols = ["arrival_date"] + (["hotel"] if hotel_col else []) + ["n", "exp", "var"]
        return pd.DataFrame(columns=cols)
    p = pd.to_numeric(scored_window["cancel_proba"], errors="coerce").fillna(0.0)
    df = pd.DataFrame({
        "arrival_date": pd.to_datetime(scored_window["arrival"], utc=True).dt.date,
        "p": p.to_numpy(),
    })
    if hotel_col and hotel_col in scored_window.columns:
        df["hotel"] = scored_window[hotel_col].to_numpy()
    df["var"] = df["p"] * (1.0 - df["p"])
    keys = ["arrival_date"] + (["hotel"] if "hotel" in df.columns else [])
    return (df.groupby(keys)
              .agg(n=("p", "size"), exp=("p", "sum"), var=("var", "sum"))
              .reset_index())


# ---- Room-type occupancy over the window (single property) -----------------
def room_type_occupancy(property_name: str,
                        today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Long frame [date, unitGroup, occupied] = # occupying bookings per room type
    per night over the window, for one property.

    Occupancy counts bookings that OCCUPY the night (status Confirmed/InHouse, not
    cancelled/no-show/checked-out) and whose stay overlaps the night. Reads the
    reservations cache only. Empty frame if no cache / no data.
    """
    df = _reservations_cached()
    if df.empty or not {"property_name", "unitGroup_name", "arrival", "departure", "status"} <= set(df.columns):
        return pd.DataFrame(columns=["date", "unitGroup", "occupied"])
    start, end = window_bounds(today)
    sub = df[(df["property_name"] == property_name) & (df["status"].isin(OCCUPYING_STATUSES))].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "unitGroup", "occupied"])
    # Keep everything as tz-aware pandas Series (UTC) and compare with pandas — a
    # tz-aware Series.to_numpy() yields object Timestamps, which breaks numpy compares.
    arr = pd.to_datetime(sub["arrival"], utc=True).dt.normalize()
    dep = pd.to_datetime(sub["departure"], utc=True).dt.normalize()
    ug = sub["unitGroup_name"]
    nights = pd.date_range(start, end - pd.Timedelta(days=1), freq="D", tz="UTC")
    records = []
    for night in nights:
        occ_mask = (arr <= night) & (dep > night)   # arrived on/before night, departs after
        if occ_mask.any():
            for group, cnt in ug[occ_mask].value_counts().items():
                records.append({"date": night, "unitGroup": group, "occupied": int(cnt)})
    return pd.DataFrame(records, columns=["date", "unitGroup", "occupied"])
