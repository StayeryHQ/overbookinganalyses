# dash_app/backend/data_access.py
# Read-only accessors the Occupancy dashboard uses. EVERYTHING here reads from the
# Phase-1 local caches (parquet) — no live BigQuery is ever triggered by a filter or
# table interaction (hard performance requirement). Re-scoring the upcoming window is
# a separate write path that runs on the file-backed job runner (see
# model_ops.score_window_job), never inline on a filter.

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
# Default DECISION threshold when no costs are entered yet: the analytic cost-optimal
# point for the project's default walk/empty costs. Everything downstream (heatmap
# count, KPI, table Low/Medium boundary) uses the cost-based threshold instead once
# costs are set — see cost_optimal_threshold().
DEFAULT_RISK_THRESHOLD = float(sc.analytic_threshold())
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
    """LOCAL-time display string ('Jul 13, 2026, 14:30 CEST') via the one shared
    formatter src.fmt_ts_local; storage stays UTC. None if unparseable."""
    return src.fmt_ts_local(value)


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


# ---- Display enrichment: risk label + group flag ---------------------------
def add_display_columns(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Add `risk_label` (cost-based Low/Medium/High) and `is_group` (booking is
    part of a group: blockId or groupName present). No-op on an empty frame.

    `threshold` is the cost-based decision threshold — it is the Low/Medium boundary;
    High is the fixed 0.85 cutoff (src.HIGH_RISK_CUTOFF). Defaults to
    DEFAULT_RISK_THRESHOLD so the column is never blank just because no costs were
    passed. Recomputed live at display time, so changing the costs re-colours the
    table without re-scoring.
    """
    if df.empty:
        return df
    out = df.copy()
    thr = DEFAULT_RISK_THRESHOLD if threshold is None else float(threshold)
    if "cancel_proba" in out.columns:
        out["risk_label"] = [src.risk_label_cost(p, thr) for p in out["cancel_proba"]]
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


# NOTE: capacity is no longer a single per-property number. It comes PER DAY from
# _perf_daily() (houseCount straight from the perf table). The old
# _capacity_from_perf()/property_capacity() helpers were removed as dead code when the
# heatmap switched to the PMS `occupancyPercentage` (2026-07 — nothing else referenced them).


@lru_cache(maxsize=1)
def _perf_daily() -> pd.DataFrame:
    """property_performance_daily with `property_name` attached and the businessDay
    normalised to a UTC midnight `_day`.

    THE authoritative operational source for the heatmap: occupancy %, sold/house
    counts, arrivals/departures and cancellations come STRAIGHT from the PMS here —
    never re-derived from reservations. `occupancyPercentage` is already
    soldCount / (houseCount − outOfOrderCount) × 100 (so it can exceed 100 % when a
    night is overbooked); we take it verbatim. Empty frame if the cache/mapping is
    unavailable. Cached; cleared after every data refresh (see model_ops).
    """
    p = _data_dir() / PERF_CACHE_FILE
    if not p.exists():
        return pd.DataFrame()
    perf = pd.read_parquet(p)
    if perf.empty or "propertyId" not in perf.columns:
        return pd.DataFrame()
    code2name = _property_code_to_name()
    out = perf.copy()
    out["property_name"] = out["propertyId"].astype(str).map(code2name)
    out["_day"] = pd.to_datetime(out["businessDay"], utc=True, errors="coerce").dt.normalize()
    return out.dropna(subset=["property_name", "_day"])


# ---- ONE cost-based decision threshold (shared by every view) --------------
def cost_optimal_threshold(walk: float | None, empty: float | None,
                           high_demand: bool = False,
                           multiplier: float | None = None,
                           model: str | None = None) -> float:
    """The single cost-optimal DECISION threshold from the entered walk/empty costs.

    Applies the high-demand multiplier to the walk cost first (effective walk cost),
    then uses the model's cost-minimising validation point (sc.operating_threshold),
    falling back to the analytic Bayes value if validation predictions are missing.
    Clamped to [0, 1] so negative or degenerate costs can never produce an
    out-of-range threshold. This is what drives the heatmap count, the KPI, and the
    table's Low/Medium boundary.
    """
    w = sc.COST_WALK if walk in (None, "") else float(walk)
    e = sc.COST_EMPTY if empty in (None, "") else float(empty)
    mult = src.DEFAULT_HIGH_DEMAND_MULTIPLIER if multiplier in (None, "") else float(multiplier)
    eff_walk = src.effective_walk_cost(w, bool(high_demand), mult)
    try:
        thr = sc.operating_threshold(sc.resolve_model(model), eff_walk, e)
    except Exception:  # noqa: BLE001 — no artifact/validation preds -> analytic Bayes
        thr = sc.analytic_threshold(eff_walk, e)
    return float(min(max(thr, 0.0), 1.0))

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


def empty_room_cost_prefill_global() -> tuple[float | None, str]:
    """One global empty-room cost pre-fill: the average of the per-property values
    from empty_room_cost_prefill(). None if no source is available (then the RM just
    enters it). Used because the costs are now a single global setting."""
    by_name, source = empty_room_cost_prefill()
    if not by_name:
        return None, source
    val = round(float(sum(by_name.values()) / len(by_name)), 2)
    return val, f"{source}, avg across properties"


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
    """Per (property_name, day) over the 14-day window.

    OPERATIONAL columns come STRAIGHT from the PMS property_performance_daily table
    (no re-derivation): occupancy_pct = `occupancyPercentage` (already
    soldCount/(houseCount−outOfOrder)×100, so it can exceed 100 % = overbooked),
    occupied_units = `soldCount`, capacity = `houseCount`, out_of_order =
    `outOfOrderCount`, arrivals = `arrivalsCount`, departures = `departuresCount`,
    cancellations = `cancellationsCount`.

    MODEL columns come from the scored upcoming arrivals (reservations): exp_cancels =
    Σ P(cancel) over that night's arrivals (threshold-independent) and pred_cancels =
    COUNT of arrivals with cancel_proba ≥ the cost threshold. NaN occupancy_pct means
    the perf table has no row for that (property, day).
    """
    thr = DEFAULT_RISK_THRESHOLD if threshold is None else float(threshold)
    start, end = window_bounds(today)
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D", tz="UTC")
    props = properties or property_list()

    # Base grid: every (property, day), keeping the given property order.
    grid = pd.DataFrame([(p, d) for p in props for d in days],
                        columns=["property_name", "_day"])

    # --- OPERATIONAL truth: taken verbatim from the PMS perf table ---------------
    perf = _perf_daily()
    if not perf.empty:
        perf = perf[perf["property_name"].isin(props)]
        rename = {"occupancyPercentage": "occupancy_pct", "soldCount": "occupied_units",
                  "houseCount": "capacity", "outOfOrderCount": "out_of_order",
                  "arrivalsCount": "arrivals", "departuresCount": "departures",
                  "cancellationsCount": "cancellations"}
        keep = ["property_name", "_day"] + [c for c in rename if c in perf.columns]
        grid = grid.merge(perf[keep].rename(columns=rename),
                          on=["property_name", "_day"], how="left")

    # --- MODEL outputs: over the night's scored arrivals (reservations) ----------
    scored = load_scored()
    if not scored.empty and "property_name" in scored.columns:
        scored = scored[scored["property_name"].isin(props)]
    if not scored.empty and "cancel_proba" in scored.columns:
        cp = pd.to_numeric(scored["cancel_proba"], errors="coerce")
        s_all = pd.to_datetime(scored["arrival"], utc=True).dt.normalize()
        exp_df = (pd.DataFrame({"property_name": scored["property_name"].to_numpy(),
                                "_day": s_all.to_numpy(),
                                "exp_cancels": cp.fillna(0.0).to_numpy()})
                  .groupby(["property_name", "_day"])["exp_cancels"].sum().reset_index())
        grid = grid.merge(exp_df, on=["property_name", "_day"], how="left")
        hot = scored[cp >= thr]
        if not hot.empty:
            s_arr = pd.to_datetime(hot["arrival"], utc=True).dt.normalize()
            pred_df = (pd.DataFrame({"property_name": hot["property_name"].to_numpy(),
                                     "_day": s_arr.to_numpy()})
                       .groupby(["property_name", "_day"]).size()
                       .rename("pred_cancels").reset_index())
            grid = grid.merge(pred_df, on=["property_name", "_day"], how="left")

    for col in ("occupied_units", "capacity", "out_of_order", "arrivals", "departures",
                "cancellations", "pred_cancels"):
        grid[col] = (pd.to_numeric(grid[col], errors="coerce").fillna(0).astype(int)
                     if col in grid.columns else 0)
    grid["exp_cancels"] = (pd.to_numeric(grid["exp_cancels"], errors="coerce").fillna(0.0)
                           if "exp_cancels" in grid.columns else 0.0)
    # occupancy_pct is taken as-is from the table (already a %, may exceed 100). Left NaN
    # where the perf table has no row, so the heatmap shows a blank tile, never a fake 0 %.
    grid["occupancy_pct"] = (pd.to_numeric(grid["occupancy_pct"], errors="coerce").round(1)
                             if "occupancy_pct" in grid.columns else float("nan"))
    grid["day"] = grid["_day"].dt.date.astype(str)
    return grid[["property_name", "day", "occupancy_pct", "occupied_units",
                 "capacity", "out_of_order", "arrivals", "departures", "cancellations",
                 "pred_cancels", "exp_cancels"]]
