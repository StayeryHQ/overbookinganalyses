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
PERF_CACHE_FILE = "property_performance_daily.parquet"

# Statuses that OCCUPY a room on a given night (for the room-type occupancy view).
OCCUPYING_STATUSES = ("Confirmed", "InHouse")
# Statuses excluded EVERYWHERE (scoring, KPIs, heatmap, table). A cancelled booking
# has zero bearing on future occupancy/risk. Filtered once here, at the data layer.
EXCLUDED_STATUSES = ("Canceled",)


def _data_dir() -> Path:
    return src.data_dir()


def _drop_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """Remove already-cancelled bookings. Single choke point so no chart can forget."""
    if df.empty or "status" not in df.columns:
        return df
    return df[~df["status"].astype("string").isin(EXCLUDED_STATUSES)].copy()


def _fmt_ts(value) -> str | None:
    """Human-readable timestamp: 'Jul 02, 2026, 14:30' (no milliseconds, no offset).
    Parses ISO strings / datetimes via pandas. Returns None if unparseable/empty."""
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%b %d, %Y, %H:%M")


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
    dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return _fmt_ts(dt)   # clean 'Jul 02, 2026, 14:30' (no ms / offset)


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
            out["retrained_at"] = _fmt_ts(card.get("retrained_at"))
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
    """Read the cached scored upcoming bookings, already-cancelled bookings removed.
    Empty frame if not scored yet."""
    p = _data_dir() / SCORED_CACHE_FILE
    if not p.exists():
        return pd.DataFrame()
    return _drop_cancelled(pd.read_parquet(p))


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
    """Per-(arrival-night[, hotel]) expected freed rooms from the scored window.
    Thin wrapper over src.hazard.per_night_table — ONE implementation of
    exp = Σp / var = Σp(1-p), instead of a drifting inline copy."""
    if scored_window.empty or "cancel_proba" not in scored_window.columns:
        cols = ["arrival_date"] + (["hotel"] if hotel_col else []) + ["n", "exp", "var"]
        return pd.DataFrame(columns=cols)
    from src.hazard import per_night_table
    p = pd.to_numeric(scored_window["cancel_proba"], errors="coerce").fillna(0.0)
    return per_night_table(scored_window, p.to_numpy(), hotel_col=hotel_col)


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


# ---- Display enrichment: risk label + group flag ---------------------------
def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `risk_label` (config-driven Low/Medium/High) and `is_group` (booking is
    part of a group: blockId or groupName present). No-op on an empty frame."""
    if df.empty:
        return df
    out = df.copy()
    if "cancel_proba" in out.columns:
        out["risk_label"] = [src.risk_label(p) for p in out["cancel_proba"]]
    else:
        out["risk_label"] = ""

    def _txt(col: str) -> pd.Series:
        """Column as string Series, '' for missing values AND missing columns —
        so a schema drift in ONE of the two group fields can't crash the page."""
        if col not in out.columns:
            return pd.Series([""] * len(out), index=out.index, dtype="string")
        return out[col].astype("string").fillna("")

    out["is_group"] = (_txt("blockId").str.len() > 0) | (_txt("groupName").str.len() > 0)
    return out


# ---- Capacity per property (for occupancy %) -------------------------------
@lru_cache(maxsize=1)
def _property_code_to_name() -> dict[str, str]:
    """{property_code -> property_name} from the reservations cache — the bridge
    between the performance table's propertyId (e.g. 'BER_FR') and the
    property_name used everywhere else. Empty dict if the columns are absent."""
    df = _reservations_cached()
    if df.empty or not {"property_code", "property_name"} <= set(df.columns):
        return {}
    m = df[["property_code", "property_name"]].dropna().drop_duplicates()
    return dict(zip(m["property_code"].astype(str), m["property_name"].astype(str)))


@lru_cache(maxsize=1)
def _capacity_from_perf() -> dict[str, int]:
    """{property_name -> total bookable units} from the performance table's most recent
    houseCount, mapped propertyId->property_name via the reservations cache. Real data,
    not a placeholder. Empty dict if the perf cache or the mapping is unavailable."""
    p = _data_dir() / PERF_CACHE_FILE
    if not p.exists():
        return {}
    perf = pd.read_parquet(p)
    if perf.empty or not {"propertyId", "houseCount"} <= set(perf.columns):
        return {}
    if "businessDay" in perf.columns:
        perf = perf.assign(_bd=pd.to_datetime(perf["businessDay"], errors="coerce")).sort_values("_bd")
    latest = perf.groupby("propertyId")["houseCount"].last()   # current room count
    code2name = _property_code_to_name()
    out: dict[str, int] = {}
    for code, hc in latest.items():
        name = code2name.get(str(code))
        if name and pd.notna(hc) and hc > 0:
            out[name] = int(hc)
    return out


def property_capacity() -> dict[str, int]:
    """Total bookable units per property_name, for the occupancy-% heatmap.

    Primary source is the performance table's houseCount (real room counts), mapped to
    property_name via the reservations cache's property_code. Falls back to the sum of
    the hand-maintained room-type capacities in configs/room_type_capacity.yaml (unset
    today). Returns {} only if neither is available — the heatmap then shows occupied
    units without a % (never a fabricated %).
    """
    caps = _capacity_from_perf()
    if caps:
        return caps
    yaml_caps = src.load_room_type_capacity()   # fallback: hand-maintained room types
    return {prop: int(sum(groups.values())) for prop, groups in yaml_caps.items() if groups}


# ---- Empty-room cost pre-fill (visible in the input) -----------------------
def empty_room_cost_prefill() -> tuple[dict[str, float], str]:
    """({property_name: value}, source_label) to PRE-FILL the empty-room cost.

    Preferred source: the property's real average daily rate (ADR) from the
    performance table, keyed to property_name via the reservations cache's
    property_code. Fallback: average gross-per-night from reservations. The
    source label is shown in the UI so the RM knows what the number means.
    """
    adr = src.average_room_rate_by_property()      # {propertyId: adr, last 90 days}
    code2name = _property_code_to_name()
    if adr and code2name:
        by_name = {code2name[c]: v for c, v in adr.items() if c in code2name}
        if by_name:
            return by_name, "avg. daily rate (ADR, last 90 days)"
    # Fallback proxy: average gross per night per property_name from reservations.
    res = _reservations_cached()
    if res.empty or "property_name" not in res.columns:
        return {}, "unavailable"
    gross = pd.to_numeric(res.get("totalGrossAmount_amount"), errors="coerce")
    arr = pd.to_datetime(res["arrival"], utc=True, errors="coerce")
    dep = pd.to_datetime(res["departure"], utc=True, errors="coerce")
    nights = ((dep.dt.normalize() - arr.dt.normalize()) / pd.Timedelta(days=1)).clip(lower=1)
    gpn = (gross / nights).replace([float("inf"), -float("inf")], pd.NA)
    tmp = pd.DataFrame({"property_name": res["property_name"], "gpn": gpn}).dropna()
    if tmp.empty:
        return {}, "unavailable"
    means = tmp.groupby("property_name")["gpn"].mean()
    return {str(k): round(float(v), 2) for k, v in means.items()}, "avg. room revenue / night (proxy)"


# ---- Arrivals filtering (for composition charts + table) -------------------
def arrivals_window(scored: pd.DataFrame, properties: list[str] | None = None,
                    day: str | None = None, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Scored bookings whose ARRIVAL falls in the window (and, if given, on the
    exact `day`) for the selected properties. `day` is an ISO date string
    (YYYY-MM-DD) from a heatmap tile click; None = all 14 days (aggregate)."""
    win = in_window(scored, properties, today=today)
    if day is not None and not win.empty:
        d = pd.to_datetime(win["arrival"], utc=True).dt.normalize()
        target = pd.Timestamp(day, tz="UTC").normalize()
        win = win[d == target].copy()
    return win


# ---- Heatmap grid: one row per (property, day) -----------------------------
def heatmap_grid(properties: list[str] | None = None, threshold: float | None = None,
                 today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per (property_name, day) over the 14-day window:
    occupancy_pct (NaN if capacity unknown), occupied_units, arrivals, departures,
    pred_cancels (scored arrivals with cancel_proba >= threshold). Cancelled bookings
    are already excluded (reservations filtered here, scored filtered in load_scored).
    """
    thr = DEFAULT_RISK_THRESHOLD if threshold is None else float(threshold)
    start, end = window_bounds(today)
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D", tz="UTC")
    props = properties or property_list()

    res = _drop_cancelled(_reservations_cached())
    scored = load_scored()
    caps = property_capacity()

    rows = []
    for prop in props:
        sub = res[res["property_name"] == prop] if not res.empty else res
        a = pd.to_datetime(sub["arrival"], utc=True).dt.normalize() if not sub.empty else pd.Series([], dtype="datetime64[ns, UTC]")
        d = pd.to_datetime(sub["departure"], utc=True).dt.normalize() if not sub.empty else pd.Series([], dtype="datetime64[ns, UTC]")
        occ_ok = sub["status"].isin(OCCUPYING_STATUSES) if not sub.empty else pd.Series([], dtype=bool)
        sc_sub = scored[scored["property_name"] == prop] if not scored.empty else scored
        sc_arr = (pd.to_datetime(sc_sub["arrival"], utc=True).dt.normalize()
                  if not sc_sub.empty else None)
        sc_p = (pd.to_numeric(sc_sub["cancel_proba"], errors="coerce")
                if not sc_sub.empty else None)
        cap = caps.get(prop)
        for day in days:
            arrivals = int((a == day).sum()) if len(a) else 0
            departures = int((d == day).sum()) if len(d) else 0
            occupied = int(((a <= day) & (d > day) & occ_ok).sum()) if len(a) else 0
            occ_pct = (round(occupied / cap * 100, 1) if cap else float("nan"))
            pred = int(((sc_arr == day) & (sc_p >= thr)).sum()) if sc_arr is not None else 0
            rows.append({"property_name": prop, "day": day.date().isoformat(),
                         "occupancy_pct": occ_pct, "occupied_units": occupied,
                         "capacity": cap, "arrivals": arrivals,
                         "departures": departures, "pred_cancels": pred})
    return pd.DataFrame(rows)
