# dash_app/backend/cancellation_history.py
# Read-only aggregators for Page 3 (Cancellation History). EVERYTHING here reads the
# cleaned reservations cache (Data/reservations_clean.parquet)  the same file the
# models train on  via src.load_clean_reservations(). No live BigQuery, ever.
#
# CORRECTNESS NOTE (differs from the Occupancy page):
#   On this HISTORICAL page a cancellation is the thing we MEASURE, so cancelled
#   bookings are the NUMERATOR and must NOT be dropped. `status` in the clean cache is
#   the encoded target  1 = cancel-before-arrival (positive class), 0 = stayed
#   (CheckedOut/NoShow/InHouse that did not cancel before arrival). Base rate ≈ 0.198.
#   (The Occupancy page excludes cancelled bookings because it is forward-looking; that
#   rule is deliberately NOT applied here.)
#
# Every public function accepts an optional `properties` filter (None/empty => all 11
# locations) so a single global filter drives every chart on the page. Functions return
# small, already-aggregated frames (server-side aggregation  never ship raw rows to the
# client). Rates computed over too few bookings are masked to NaN (min-sample guards
# below) rather than drawn as confident signal  matches the experiment notebooks.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Column / bucket vocabulary -------------------------------------------
TARGET = "status"                       # int8: 1 = cancel-before-arrival, 0 = stayed
STAY_ORDER = ["short", "mid", "long"]   # canonical order of the precomputed stay_bucket
# Real boundaries verified against the cache (los_nights per bucket): short 1–2,
# mid 3–6, long 7+. Surfaced in tooltips so "stay segment" is never ambiguous.
STAY_LABELS = {"short": "Short · 1–2 nights",
               "mid": "Mid · 3–6 nights",
               "long": "Long · 7+ nights"}
LEAD_BINS = [-1, 7, 30, 90, np.inf]     # lead_time_days is fractional; -1 keeps 0 in bin 1
LEAD_LABELS = ["0–7 d", "8–30 d", "31–90 d", "90 d+"]

# ---- Min-sample guards (a rate over too few bookings is noise, not signal) --
MIN_N_MONTH = 50        # monthly line points (matches experiments/cancellation_rate_over_time)
MIN_N_CELL = 30         # property × month heatmap cells
MIN_N_CHANNEL = 200     # channel-deviation bars
MIN_N_KPI = 100         # "highest-rate location" KPI guard
MIN_N_DAILY = 50        # per-day lead / per-night stay points (daily-granularity charts)


# ---- Loading / preparation -------------------------------------------------
def _load_clean_df() -> pd.DataFrame:
    """Load the cleaned reservations cache. Prefers the canonical src loader (same one
    the notebooks/models use); falls back to reading the parquet by repo-relative path
    so the pure aggregators remain usable in a minimal environment (e.g. unit tests)."""
    try:
        from src import load_clean_reservations
        return load_clean_reservations()
    except Exception:  # noqa: BLE001  heavy src deps may be unavailable; use the file.
        p = Path(__file__).resolve().parents[2] / "Data" / "reservations_clean.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns every aggregator needs (UTC-safe arrival month, numeric
    target). Returns a copy; safe on an empty frame."""
    if df.empty:
        return df
    out = df.copy()
    arr = pd.to_datetime(out["arrival"], utc=True)
    out["arrival"] = arr
    # tz_localize(None) strips the UTC offset (keeping the UTC wall time) so to_period
    # doesn't warn about dropping tz; month = first-of-month timestamp.
    out["month"] = arr.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out


@lru_cache(maxsize=1)
def _clean() -> pd.DataFrame:
    """Prepared clean-reservations frame, cached per process."""
    return _prepare(_load_clean_df())


def _filtered(properties: list[str] | None,
              window_months: int | None = None) -> pd.DataFrame:
    """Clean frame limited to the selected properties (None/empty => all) and,
    optionally, to the last `window_months` months of arrivals (None => full history).
    This is the single choke point the page's global location + time-window filters both
    flow through, so every chart stays consistent. The window is anchored on the newest
    month in the FULL cache, so it means the same span regardless of the location selection."""
    df = _clean()
    if df.empty:
        return df
    if properties:
        df = df[df["property_name"].isin(properties)]
    if window_months:
        last = _clean()["month"].max()
        start = (last.to_period("M") - (int(window_months) - 1)).to_timestamp()
        df = df[df["month"] >= start]
    return df


@lru_cache(maxsize=1)
def property_list() -> list[str]:
    """The 11 property names (sorted) from the clean cache. Verified identical to the
    Occupancy page's list, so the location filter is consistent across pages."""
    df = _clean()
    if df.empty or "property_name" not in df.columns:
        return []
    return sorted(df["property_name"].dropna().unique().tolist())


def base_rate() -> float | None:
    """Global cancel rate across ALL locations (the fixed reference line)."""
    df = _clean()
    return float(df[TARGET].mean()) if not df.empty else None


def selection_rate(properties: list[str] | None = None,
                   window_months: int | None = None) -> float | None:
    """Overall cancel rate for the current selection (the dashed reference on the
    breakdown charts). Cheaper than kpis() when only this one number is needed."""
    df = _filtered(properties, window_months)
    return float(df[TARGET].mean()) if not df.empty else None


def date_span() -> tuple[str, str] | None:
    """('Aug 2022', 'Jun 2026') month labels of the available history, or None."""
    df = _clean()
    if df.empty:
        return None
    return df["month"].min().strftime("%b %Y"), df["month"].max().strftime("%b %Y")


def _rate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Group by `keys` and return n + cancel_rate (mean of the 0/1 target)."""
    return (df.groupby(keys)[TARGET]
              .agg(["size", "mean"]).reset_index()
              .rename(columns={"size": "n", "mean": "cancel_rate"}))


# ---- 1) Cancellation rate over time (monthly) ------------------------------
def _monthly(df: pd.DataFrame, per_property: bool = False) -> pd.DataFrame:
    """[month(, property_name), n, cancel_rate]. Rates on months with n < MIN_N_MONTH
    are masked to NaN (kept as rows so `n` still feeds tooltips)."""
    cols = ["month"] + (["property_name"] if per_property else []) + ["n", "cancel_rate"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    keys = ["month"] + (["property_name"] if per_property else [])
    g = _rate(df, keys)
    g["cancel_rate"] = g["cancel_rate"].where(g["n"] >= MIN_N_MONTH)
    return g.sort_values(keys).reset_index(drop=True)


def monthly_rate(properties: list[str] | None = None, per_property: bool = False,
                 window_months: int | None = None) -> pd.DataFrame:
    return _monthly(_filtered(properties, window_months), per_property)


# ---- 2) Property × month heatmap matrix ------------------------------------
def _property_month(df: pd.DataFrame, months_back: int = 12,
                    min_n: int = MIN_N_CELL) -> pd.DataFrame:
    """[property_name, month, n, cancel_rate] over the last `months_back` months.
    Cells with n < min_n are masked to NaN (rendered as blank/grey)."""
    cols = ["property_name", "month", "n", "cancel_rate"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    last = df["month"].max()
    start = (last.to_period("M") - (months_back - 1)).to_timestamp()
    sub = df[df["month"] >= start]
    g = _rate(sub, ["property_name", "month"])
    g["cancel_rate"] = g["cancel_rate"].where(g["n"] >= min_n)
    return g.sort_values(["property_name", "month"]).reset_index(drop=True)


def property_month_matrix(properties: list[str] | None = None, months_back: int = 12,
                          window_months: int | None = None) -> pd.DataFrame:
    # `months_back` = how many months the heatmap DISPLAYS; `window_months` = the page's
    # global time filter applied first. The heatmap shows whichever is the tighter span.
    return _property_month(_filtered(properties, window_months), months_back)


def flag_anomalies(properties: list[str] | None = None, months_back: int = 12,
                   factor: float = 1.5, window_months: int | None = None) -> pd.DataFrame:
    """Property-months whose cancel rate exceeds the global base rate by >= `factor`×
    (and clears the min-sample guard). Drives the 'unusually high' warnings/badges."""
    g = property_month_matrix(properties, months_back, window_months).dropna(subset=["cancel_rate"])
    base = _clean()[TARGET].mean() if not _clean().empty else np.nan
    if g.empty or not np.isfinite(base):
        return g.assign(base_rate=base).iloc[0:0]
    hot = g[g["cancel_rate"] >= factor * base].copy()
    hot["base_rate"] = base
    return hot.sort_values("cancel_rate", ascending=False).reset_index(drop=True)


# ---- 3) Channel: deviation from the base rate ------------------------------
def _channel_dev(df: pd.DataFrame, min_n: int = MIN_N_CHANNEL) -> tuple[pd.DataFrame, float]:
    """([channel, n, cancel_rate, deviation], base_rate). `deviation` = channel rate −
    base rate of the current selection. Channels below min_n are dropped (too thin to
    read); the long tail is therefore excluded rather than fabricated."""
    cols = ["channel", "n", "cancel_rate", "deviation"]
    if df.empty or "channelCode" not in df.columns:
        return pd.DataFrame(columns=cols), float("nan")
    base = float(df[TARGET].mean())
    g = (df.groupby("channelCode")[TARGET].agg(["size", "mean"]).reset_index()
           .rename(columns={"channelCode": "channel", "size": "n", "mean": "cancel_rate"}))
    g = g[g["n"] >= min_n].copy()
    g["deviation"] = g["cancel_rate"] - base
    return g.sort_values("deviation").reset_index(drop=True), base


def channel_deviation(properties: list[str] | None = None,
                      window_months: int | None = None) -> tuple[pd.DataFrame, float]:
    return _channel_dev(_filtered(properties, window_months))


# ---- 4a) Stay-segment (length of stay) rate --------------------------------
def _stay_segment(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["stay_bucket", "label", "n", "cancel_rate"]
    if df.empty or "stay_bucket" not in df.columns:
        return pd.DataFrame(columns=cols)
    g = _rate(df, ["stay_bucket"])
    g["order"] = g["stay_bucket"].map({k: i for i, k in enumerate(STAY_ORDER)})
    g["label"] = g["stay_bucket"].map(STAY_LABELS)
    return g.sort_values("order").drop(columns="order").reset_index(drop=True)


def stay_segment_rate(properties: list[str] | None = None,
                      window_months: int | None = None) -> pd.DataFrame:
    return _stay_segment(_filtered(properties, window_months))


# ---- 4b) Lead-time bucket rate ---------------------------------------------
def _leadtime(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["bucket", "n", "cancel_rate"]
    if df.empty or "lead_time_days" not in df.columns:
        return pd.DataFrame(columns=cols)
    lead = pd.to_numeric(df["lead_time_days"], errors="coerce")
    bucket = pd.cut(lead, bins=LEAD_BINS, labels=LEAD_LABELS)
    tmp = pd.DataFrame({"bucket": bucket, TARGET: df[TARGET].to_numpy()})
    g = (tmp.groupby("bucket", observed=True)[TARGET].agg(["size", "mean"]).reset_index()
           .rename(columns={"size": "n", "mean": "cancel_rate"}))
    g["bucket"] = pd.Categorical(g["bucket"], categories=LEAD_LABELS, ordered=True)
    return g.sort_values("bucket").reset_index(drop=True)


def leadtime_bucket_rate(properties: list[str] | None = None,
                         window_months: int | None = None) -> pd.DataFrame:
    return _leadtime(_filtered(properties, window_months))


# ---- 4c) Length-of-stay per NIGHT (daily granularity, coloured by segment) --
def _stay_segment_of(night: int) -> str:
    """Which short/mid/long bucket a night count belongs to (colour only)."""
    return "short" if night <= 2 else "mid" if night <= 6 else "long"


def _stay_daily(df: pd.DataFrame, max_night: int = 14,
                min_n: int = MIN_N_DAILY) -> pd.DataFrame:
    """[night, label, segment, n, cancel_rate]  one row per exact stay length in nights.
    Stays of >= max_night nights are pooled into a single 'max_night+' bin (individually
    too thin). `segment` is the short(1–2)/mid(3–6)/long(7+) bucket the length falls in,
    used only to colour the bars. Rates over < min_n bookings are masked to NaN."""
    cols = ["night", "label", "segment", "n", "cancel_rate"]
    if df.empty or "los_nights" not in df.columns:
        return pd.DataFrame(columns=cols)
    los = pd.to_numeric(df["los_nights"], errors="coerce").astype("Int64")
    night = los.clip(lower=1, upper=max_night)
    tmp = pd.DataFrame({"night": night, TARGET: df[TARGET].to_numpy()}).dropna(subset=["night"])
    g = (tmp.groupby("night")[TARGET].agg(["size", "mean"]).reset_index()
           .rename(columns={"size": "n", "mean": "cancel_rate"}))
    g["night"] = g["night"].astype(int)
    g["cancel_rate"] = g["cancel_rate"].where(g["n"] >= min_n)
    g["segment"] = g["night"].map(_stay_segment_of)
    g["label"] = g["night"].map(lambda x: f"{x}+" if x >= max_night else str(x))
    return g.sort_values("night").reset_index(drop=True)


def stay_daily_rate(properties: list[str] | None = None, max_night: int = 14,
                    window_months: int | None = None) -> pd.DataFrame:
    return _stay_daily(_filtered(properties, window_months), max_night)


# ---- 4d) Lead-time per DAY (daily granularity, optional length-of-stay split) --
def _leadtime_daily(df: pd.DataFrame, by_stay: bool = False, max_day: int = 45,
                    min_n: int = MIN_N_DAILY) -> pd.DataFrame:
    """[lead_day(, stay_bucket), n, cancel_rate]  one row per whole lead-time day
    0..max_day. `lead_time_days` is fractional, rounded to the nearest whole day; days
    beyond max_day are dropped (the near term is the story  default 45 days). When
    by_stay is set the rate is additionally split by the short/mid/long stay bucket
    (three series). Points over < min_n bookings are masked to NaN."""
    cols = ["lead_day"] + (["stay_bucket"] if by_stay else []) + ["n", "cancel_rate"]
    if df.empty or "lead_time_days" not in df.columns:
        return pd.DataFrame(columns=cols)
    day = pd.to_numeric(df["lead_time_days"], errors="coerce").round()
    tmp = pd.DataFrame({"lead_day": day, TARGET: df[TARGET].to_numpy()})
    if by_stay:
        tmp["stay_bucket"] = df["stay_bucket"].to_numpy()
    tmp = tmp[(tmp["lead_day"] >= 0) & (tmp["lead_day"] <= max_day)].dropna(subset=["lead_day"])
    keys = ["lead_day"] + (["stay_bucket"] if by_stay else [])
    g = (tmp.groupby(keys)[TARGET].agg(["size", "mean"]).reset_index()
           .rename(columns={"size": "n", "mean": "cancel_rate"}))
    g["lead_day"] = g["lead_day"].astype(int)
    g["cancel_rate"] = g["cancel_rate"].where(g["n"] >= min_n)
    return g.sort_values(keys).reset_index(drop=True)


def leadtime_daily_rate(properties: list[str] | None = None, by_stay: bool = False,
                        max_day: int = 45, window_months: int | None = None) -> pd.DataFrame:
    return _leadtime_daily(_filtered(properties, window_months), by_stay, max_day)


# ---- 5) Cancel-timing curve (WHEN do cancellations land?) ------------------
def _cancel_timing(df: pd.DataFrame, max_days: int = 90) -> tuple[pd.DataFrame, int]:
    """([days_before, cum_share_within], n_cancellations).

    Among cancelled bookings only, using the real `cancel_days_before_arrival` field:
    cum_share_within[d] = fraction of cancellations that occurred within d days before
    arrival. Monotonic 0→1. Answers "how late can rooms still free up?"  e.g. the
    median cancellation lands ≈7 days out, so half of all freed rooms appear inside the
    final week. Cancellations further out than `max_days` collapse into the last point.
    """
    cols = ["days_before", "cum_share_within"]
    if df.empty or "cancel_days_before_arrival" not in df.columns:
        return pd.DataFrame(columns=cols), 0
    c = pd.to_numeric(df.loc[df[TARGET] == 1, "cancel_days_before_arrival"], errors="coerce").dropna()
    c = c[c >= 0]
    n = int(len(c))
    if n == 0:
        return pd.DataFrame(columns=cols), 0
    grid = np.arange(0, max_days + 1)
    cum = np.array([(c <= d).mean() for d in grid], dtype="float64")
    return pd.DataFrame({"days_before": grid, "cum_share_within": cum}), n


def cancel_timing_curve(properties: list[str] | None = None, max_days: int = 90,
                        window_months: int | None = None) -> tuple[pd.DataFrame, int]:
    return _cancel_timing(_filtered(properties, window_months), max_days)


# ---- 6) Lead × stay cancel-rate grid (the "blend" heatmap) ------------------
def _leadtime_stay_grid(df: pd.DataFrame, min_n: int = MIN_N_CELL) -> pd.DataFrame:
    """[stay_bucket, lead_bucket, n, cancel_rate]  cancel rate for every lead-bucket ×
    stay-bucket cell. Cells over < min_n bookings are masked to NaN. Feeds the heatmap
    that shows how lead time and length of stay COMBINE to drive cancellation."""
    cols = ["stay_bucket", "lead_bucket", "n", "cancel_rate"]
    if df.empty or "lead_time_days" not in df.columns or "stay_bucket" not in df.columns:
        return pd.DataFrame(columns=cols)
    lead = pd.to_numeric(df["lead_time_days"], errors="coerce")
    lb = pd.cut(lead, bins=LEAD_BINS, labels=LEAD_LABELS)
    tmp = pd.DataFrame({"stay_bucket": df["stay_bucket"].to_numpy(), "lead_bucket": lb,
                        TARGET: df[TARGET].to_numpy()})
    g = (tmp.groupby(["stay_bucket", "lead_bucket"], observed=True)[TARGET]
           .agg(["size", "mean"]).reset_index()
           .rename(columns={"size": "n", "mean": "cancel_rate"}))
    g["cancel_rate"] = g["cancel_rate"].where(g["n"] >= min_n)
    g["stay_bucket"] = pd.Categorical(g["stay_bucket"], categories=STAY_ORDER, ordered=True)
    g["lead_bucket"] = pd.Categorical(g["lead_bucket"], categories=LEAD_LABELS, ordered=True)
    return g.sort_values(["stay_bucket", "lead_bucket"]).reset_index(drop=True)


def leadtime_stay_grid(properties: list[str] | None = None,
                       window_months: int | None = None) -> pd.DataFrame:
    return _leadtime_stay_grid(_filtered(properties, window_months))


# ---- 7) Cancel-timing near-window heatmap (days-before × stay/lead) ---------
CANCEL_TIMING_MAX_DAY = 14      # near-arrival window shown day-by-day; rest pools into 15+


def _cancel_timing_grid(df: pd.DataFrame, dim: str = "stay",
                        max_day: int = CANCEL_TIMING_MAX_DAY,
                        min_bookings: int = 100) -> pd.DataFrame:
    """[row, day, day_order, n_cancel, n_atrisk, rate]  near-arrival cancellation HAZARD.

    For each whole day d before arrival (0 = arrival day … max_day):
      n_cancel  = bookings in the segment that cancelled exactly d days before arrival
      n_atrisk  = bookings in the segment that were STILL DUE TO ARRIVE at day d  i.e.
                  booked at least d days ahead (lead ≥ d) AND not yet cancelled by then
      rate      = n_cancel / n_atrisk

    So `rate` answers "of the bookings still due to arrive d days out, what share cancel
    that day"  a proper per-day cancel rate (a discrete hazard), NOT a share of all
    bookings. That is why a row does not sum to the segment's overall cancel rate, and why
    e.g. a 0–7 day lead bucket is simply empty past day 7 (those bookings never existed
    that far out). Rows are stay segments (dim='stay') or lead buckets (dim='lead').
    Segments with < min_bookings total bookings are dropped."""
    cols = ["row", "day", "day_order", "n_cancel", "n_atrisk", "rate"]
    if (df.empty or "cancel_days_before_arrival" not in df.columns
            or "lead_time_days" not in df.columns):
        return pd.DataFrame(columns=cols)
    d = df.reset_index(drop=True)
    if dim == "lead":
        rowkey = pd.cut(pd.to_numeric(d["lead_time_days"], errors="coerce"),
                        bins=LEAD_BINS, labels=LEAD_LABELS).astype("object")
    else:
        rowkey = d["stay_bucket"].astype("object").map(STAY_LABELS)
    rowkey = pd.Series(rowkey, index=d.index)
    lead = pd.to_numeric(d["lead_time_days"], errors="coerce").to_numpy()
    status = pd.to_numeric(d[TARGET], errors="coerce").to_numpy()
    cday = np.floor(pd.to_numeric(d["cancel_days_before_arrival"], errors="coerce").to_numpy())

    out = []
    for seg, idx in rowkey.groupby(rowkey).groups.items():
        pos = np.asarray(idx)
        Ls, sts, cds = lead[pos], status[pos], cday[pos]
        if len(Ls) < min_bookings:
            continue
        arr_L = Ls[sts == 0]                       # arrived: survive to arrival
        can_L, can_cd = Ls[sts == 1], cds[sts == 1]
        for day in range(max_day + 1):
            num = int(np.sum(can_cd == day))
            # at-risk at day = arrivals still ahead (lead ≥ day) + cancellations that are
            # still active at day (booked ≥ day ahead and cancel at day or closer to arrival)
            den = int(np.sum(arr_L >= day) + np.sum((can_cd <= day) & (can_L >= day)))
            out.append((str(seg), str(day), day, num, den,
                        (num / den) if den > 0 else np.nan))
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(out, columns=cols)


def cancel_timing_grid(properties: list[str] | None = None, dim: str = "stay",
                       window_months: int | None = None) -> pd.DataFrame:
    return _cancel_timing_grid(_filtered(properties, window_months), dim)


# ---- 8) No-shows (RAW resolved arrivals  the clean target hides them in 0) --
# A no-show is a booking that did NOT cancel before arrival but never checked in. The
# cleaned modelling cache collapses it into the "stayed" class, so no-shows are only
# visible in the raw reservations `status` string. Denominator = resolved arrivals
# (CheckedOut / NoShow / InHouse)  i.e. bookings that were actually due to arrive.
NOSHOW_ARRIVED = ("CheckedOut", "NoShow", "InHouse")


@lru_cache(maxsize=1)
def _noshow_prepared() -> pd.DataFrame:
    """Raw resolved arrivals with a no_show flag, arrival month, LoS and stay bucket."""
    p = Path(__file__).resolve().parents[2] / "Data" / "reservations_raw_no_pii.parquet"
    if not p.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(p)
    if raw.empty or "status" not in raw.columns or "arrival" not in raw.columns:
        return pd.DataFrame()
    df = raw[raw["status"].astype("string").isin(NOSHOW_ARRIVED)].copy()
    if df.empty:
        return df
    arr = pd.to_datetime(df["arrival"], utc=True, errors="coerce")
    df["arrival"] = arr
    df["month"] = arr.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    df["no_show"] = (df["status"].astype("string") == "NoShow").astype(int)
    dep = pd.to_datetime(df.get("departure"), utc=True, errors="coerce")
    los = (dep.dt.normalize() - arr.dt.normalize()) / pd.Timedelta(days=1)
    df["los_nights"] = los.clip(lower=1)
    df["stay_bucket"] = pd.cut(df["los_nights"], bins=[0, 2, 6, np.inf],
                               labels=STAY_ORDER).astype("object")
    return df[df["month"].notna()]


def _noshow_filtered(properties: list[str] | None,
                     window_months: int | None = None) -> pd.DataFrame:
    df = _noshow_prepared()
    if df.empty:
        return df
    if properties:
        df = df[df["property_name"].isin(properties)]
    if window_months:
        last = _noshow_prepared()["month"].max()
        start = (last.to_period("M") - (int(window_months) - 1)).to_timestamp()
        df = df[df["month"] >= start]
    return df


def _noshow_agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """[*keys, n, n_noshow, rate]  n = resolved arrivals (denominator), n_noshow = count,
    rate = no-show rate. Both the count AND the rate travel so every chart can show n."""
    g = (df.groupby(keys, observed=True)["no_show"].agg(["size", "sum", "mean"])
           .reset_index().rename(columns={"size": "n", "sum": "n_noshow", "mean": "rate"}))
    return g


def noshow_overall_rate(properties: list[str] | None = None,
                        window_months: int | None = None) -> float | None:
    df = _noshow_filtered(properties, window_months)
    return float(df["no_show"].mean()) if not df.empty else None


def noshow_monthly_rate(properties: list[str] | None = None,
                        window_months: int | None = None) -> pd.DataFrame:
    cols = ["month", "n", "n_noshow", "rate"]
    df = _noshow_filtered(properties, window_months)
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = _noshow_agg(df, ["month"])
    g["rate"] = g["rate"].where(g["n"] >= MIN_N_MONTH)      # thin months = noise
    return g.sort_values("month").reset_index(drop=True)


def noshow_property_month_matrix(properties: list[str] | None = None, months_back: int = 12,
                                 window_months: int | None = None) -> pd.DataFrame:
    cols = ["property_name", "month", "n", "n_noshow", "rate"]
    df = _noshow_filtered(properties, window_months)
    if df.empty:
        return pd.DataFrame(columns=cols)
    last = df["month"].max()
    start = (last.to_period("M") - (months_back - 1)).to_timestamp()
    g = _noshow_agg(df[df["month"] >= start], ["property_name", "month"])
    g["rate"] = g["rate"].where(g["n"] >= MIN_N_CELL)
    return g.sort_values(["property_name", "month"]).reset_index(drop=True)


def noshow_stay_rate(properties: list[str] | None = None,
                     window_months: int | None = None) -> pd.DataFrame:
    cols = ["stay_bucket", "label", "n", "n_noshow", "rate"]
    df = _noshow_filtered(properties, window_months)
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = _noshow_agg(df.dropna(subset=["stay_bucket"]), ["stay_bucket"])
    g["order"] = g["stay_bucket"].map({k: i for i, k in enumerate(STAY_ORDER)})
    g["label"] = g["stay_bucket"].map(STAY_LABELS)
    return g.sort_values("order").drop(columns="order").reset_index(drop=True)




# ---- Drill-down slices (feed the right-side detail Drawer) -----------------
def _month_ts(month_iso: str) -> pd.Timestamp:
    """'2026-03' -> first-of-month Timestamp (naive, matching the `month` column)."""
    return pd.Timestamp(month_iso + "-01")


def drill_property_month(property_name: str, month_iso: str) -> dict | None:
    """Detail for one heatmap cell: overall rate/n plus channel & stay breakdowns for
    that single property × month. Reads the full clean frame (a cell isn't affected by
    the page filter). None if the cell has no bookings."""
    df = _clean()
    if df.empty:
        return None
    sub = df[(df["property_name"] == property_name) & (df["month"] == _month_ts(month_iso))]
    if sub.empty:
        return None
    ch = (_rate(sub, ["channelCode"]).rename(columns={"channelCode": "channel"})
          .sort_values("n", ascending=False).head(6))
    return {
        "scope": "cell",
        "property": property_name,
        "month": _month_ts(month_iso).strftime("%b %Y"),
        "n": int(len(sub)),
        "rate": float(sub[TARGET].mean()),
        "channels": ch.to_dict("records"),
        "stays": _stay_segment(sub).to_dict("records"),
    }


def drill_month(properties: list[str] | None, month_iso: str) -> dict | None:
    """Detail for one month on the time-series: overall rate/n for the current selection
    plus a per-location ranking that month. None if the month has no bookings."""
    df = _filtered(properties)
    if df.empty:
        return None
    sub = df[df["month"] == _month_ts(month_iso)]
    if sub.empty:
        return None
    pp = _rate(sub, ["property_name"]).sort_values("cancel_rate", ascending=False)
    return {
        "scope": "month",
        "month": _month_ts(month_iso).strftime("%b %Y"),
        "n": int(len(sub)),
        "rate": float(sub[TARGET].mean()),
        "properties": pp.to_dict("records"),
    }


# ---- KPI headline numbers --------------------------------------------------
def kpis(properties: list[str] | None = None,
         window_months: int | None = None) -> dict:
    """Headline figures for the KPI strip. Any value not backed by enough real data is
    returned as None so the UI can show 'unavailable' rather than fabricate a number.
    `base_rate` stays the FULL-history global reference; everything else honours the
    location + time-window selection."""
    out = {"n_bookings": None, "overall_rate": None, "base_rate": None,
           "latest_month": None, "latest_rate": None, "delta_vs_base": None,
           "top_property": None, "top_rate": None, "span": date_span()}
    df = _filtered(properties, window_months)
    if df.empty:
        return out
    full = _clean()
    out["n_bookings"] = int(len(df))
    out["overall_rate"] = float(df[TARGET].mean())
    out["base_rate"] = float(full[TARGET].mean()) if not full.empty else None

    monthly = _monthly(df).dropna(subset=["cancel_rate"])
    if not monthly.empty:
        last = monthly.sort_values("month").iloc[-1]
        out["latest_month"] = last["month"].strftime("%b %Y")
        out["latest_rate"] = float(last["cancel_rate"])
        if out["base_rate"] is not None:
            out["delta_vs_base"] = out["latest_rate"] - out["base_rate"]

    pp = df.groupby("property_name")[TARGET].agg(["size", "mean"]).reset_index()
    pp = pp[pp["size"] >= MIN_N_KPI]
    if not pp.empty:
        top = pp.sort_values("mean").iloc[-1]
        out["top_property"] = str(top["property_name"])
        out["top_rate"] = float(top["mean"])
    return out
