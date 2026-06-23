# dash_app/backend/derive.py
# ---------------------------------------------------------------------------
# Pure-pandas derivations on a scored-bookings frame. PORTED (subset) from
# streamlit_app/backend/derive.py — only the functions the Dash "Overbooking
# Predictions" page needs: the prediction matrix/long table, the per-location
# recommendation, and the booking detail table. No Streamlit/Dash imports here,
# so this stays trivially testable and reusable for the future cancellation
# dashboard.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

# Canonical column constants.
from . import schema as S


def confirmed(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only confirmed bookings (drop already-canceled ones)."""
    # If the frame carries a status column, filter on it; otherwise pass through.
    if S.STATUS in df.columns:
        return df[df[S.STATUS] == S.STATUS_CONFIRMED]
    return df


def prediction_long(df, dates, units_by_hotel, threshold=S.HIGH_THR) -> pd.DataFrame:
    """Long (tidy) per-(hotel, date) prediction summary over the window `dates`.

    For each hotel and each date we compute: number of arrivals, expected
    cancellations (= sum of cancel probabilities), and high-risk count
    (bookings with proba >= threshold). Zero-filled where there are no arrivals.
    """
    # Restrict to confirmed bookings arriving within the window.
    df = confirmed(df)
    win = df[df[S.ARRIVAL_DATE].isin(dates)]
    # Pre-group once by (hotel, arrival date) for O(1) lookups in the loop.
    grouped = {k: g for k, g in win.groupby([S.HOTEL_CODE, S.ARRIVAL_DATE])}
    recs = []
    # Iterate every hotel x date cell so the matrix is dense (no gaps).
    for h in units_by_hotel:
        for d in dates:
            g = grouped.get((h, d))
            if g is None or g.empty:
                # No arrivals that day -> zeros.
                recs.append({"date": d, S.HOTEL_CODE: h, "arrivals": 0,
                             "expected_cancels": 0.0, "high_risk": 0})
            else:
                recs.append({"date": d, S.HOTEL_CODE: h, "arrivals": len(g),
                             "expected_cancels": float(g[S.CANCEL_PROBA].sum()),
                             "high_risk": int((g[S.CANCEL_PROBA] >= threshold).sum())})
    return pd.DataFrame.from_records(recs)


def prediction_matrix(df, dates, units_by_hotel, threshold=S.HIGH_THR,
                      value="expected_cancels") -> pd.DataFrame:
    """Pivot the long table into a hotel x date matrix of `value` (for heatmaps)."""
    # Build the tidy table, then pivot it: rows=hotel, cols=date.
    long = prediction_long(df, dates, units_by_hotel, threshold)
    return long.pivot(index=S.HOTEL_CODE, columns="date", values=value).reindex(list(units_by_hotel))


def recommendation_table(df, dates, units_by_hotel, threshold, labels) -> pd.DataFrame:
    """Per-location overbooking recommendation over the window.

    Recommendation = min(allowance, round(expected cancellations per day)) where
    the allowance is the benchmark (2 for <50 units, 4 for >=50). Mirrors the
    Streamlit logic exactly.
    """
    pl = prediction_long(df, dates, units_by_hotel, threshold)
    # Horizon length (>=1 to avoid division by zero).
    horizon = max(len(dates), 1)
    rows = []
    for h in units_by_hotel:
        g = pl[pl[S.HOTEL_CODE] == h]
        expected = float(g["expected_cancels"].sum())
        # Benchmark allowance from the units count.
        allow = 4 if units_by_hotel[h] >= 50 else 2
        per_day = expected / horizon
        # Stay within the allowance but use the statistical cancel buffer.
        rec = int(min(allow, round(per_day)))
        rows.append({
            "Standort": labels.get(h, h),
            "Units": units_by_hotel[h],
            "Anreisen": int(g["arrivals"].sum()),
            "Erwartete Stornos": round(expected, 1),
            "Ø / Tag": round(per_day, 1),
            "High-Risk": int(g["high_risk"].sum()),
            "Overbooking-Limit": allow,
            "Empfehlung": rec,
        })
    # Sort so the locations with the strongest recommendation surface first.
    return pd.DataFrame(rows).sort_values("Empfehlung", ascending=False)


def predictions_table(df, dates=None, hotel_code=None, day=None) -> pd.DataFrame:
    """Filtered, sorted per-booking table (highest cancel probability first)."""
    out = confirmed(df)
    # Optional filters: within window, single hotel, single day.
    if dates is not None:
        out = out[out[S.ARRIVAL_DATE].isin(dates)]
    if hotel_code:
        out = out[out[S.HOTEL_CODE] == hotel_code]
    if day is not None:
        out = out[out[S.ARRIVAL_DATE] == pd.Timestamp(day).normalize()]
    return out.sort_values(S.CANCEL_PROBA, ascending=False).copy()


def risk_bucket_counts(df) -> pd.DataFrame:
    """Count bookings per risk bucket (low / uncertain / high) for the bar chart.

    Returns a small frame with the German bucket label and the count, in the
    fixed low->uncertain->high order so the chart is stable.
    """
    df = confirmed(df)
    # value_counts gives a Series indexed by bucket name; reindex to fix order.
    counts = df[S.RISK_BUCKET].value_counts()
    rows = []
    for b in S.RISK_BUCKETS:
        rows.append({"bucket": b,
                     "label": S.RISK_LABELS_DE.get(b, b),
                     "count": int(counts.get(b, 0))})
    return pd.DataFrame(rows)


def daily_grid(df, dates, units_by_hotel, perf=None, threshold=S.HIGH_THR) -> pd.DataFrame:
    """Per-(hotel, date) operations grid: arrivals, departures, occupancy, expected cancels.

    This feeds the big 14-day raster heatmap. Every (hotel × date) cell is filled
    (dense grid). Definitions:
      * arrivals          = confirmed bookings ARRIVING that day,
      * departures        = confirmed bookings DEPARTING that day,
      * occupancy         = PLACEHOLDER from the booking data: bookings whose stay
                            covers that night (arrival_date <= d < departure_date)
                            divided by the property's units. The REAL occupancy will
                            come from the BigQuery occupancy table later; until then
                            this booking-derived proxy lets the dashboard be built.
      * expected_cancels  = sum of cancel probabilities of that day's arrivals
                            (risk-neutral expected freed rooms, Σ p).
      * high_conf_cancels = COUNT of that day's arrivals with cancel_proba >=
                            `threshold` — the CONSERVATIVE figure: only bookings we
                            are "fairly sure" will cancel. Driven by the UI slider.
    """
    d = confirmed(df)                                       # drop already-canceled rows

    # Build (propertyId, UTC-normalised date) -> occupancy/departures from the REAL
    # performance table where rows exist. occupancyPercentage may be 0..100 -> /100.
    occ_lk: dict = {}; dep_lk: dict = {}
    if perf is not None and len(perf) and "businessDay" in perf.columns:
        p = perf.copy()
        # NB: itertuples mangles leading-underscore names, so use a plain name.
        p["perfdate"] = pd.to_datetime(p["businessDay"], utc=True, errors="coerce").dt.normalize()
        for r in p.itertuples():
            key = (str(r.propertyId), r.perfdate)
            occv = getattr(r, "occupancyPercentage", None)
            if occv is not None and pd.notna(occv):
                occ_lk[key] = (float(occv) / 100.0) if float(occv) > 1.5 else float(occv)
            depv = getattr(r, "departuresCount", None)
            if depv is not None and pd.notna(depv):
                dep_lk[key] = int(depv)

    recs = []                                               # collected cell records
    for h in units_by_hotel:                                # every property
        hd = d[d[S.HOTEL_CODE] == h]                        # its bookings
        units = max(int(units_by_hotel[h]), 1)             # units (>=1, avoid /0)
        arr = hd[S.ARRIVAL_DATE]                            # arrival dates (reused)
        dep = hd[S.DEPARTURE_DATE]                          # departure dates (reused)
        for dt in dates:                                    # every day in the window
            # UTC-normalised key to match the (tz-aware) performance dates.
            _k = pd.Timestamp(dt)
            _k = (_k.tz_localize("UTC") if _k.tzinfo is None else _k.tz_convert("UTC")).normalize()
            key = (str(h), _k)
            arrivals   = int((arr == dt).sum())            # arriving that day (reservations)
            # Departures: prefer the performance table, else count from bookings.
            departures = dep_lk.get(key, int((dep == dt).sum()))
            # Occupancy: prefer the performance table, else booking-derived proxy
            # (bookings staying the night of `dt`: arrival<=dt<departure) / units.
            if key in occ_lk:
                occupancy = occ_lk[key]
            else:
                occupancy = int(((arr <= dt) & (dep > dt)).sum()) / units
            # Expected cancellations among that day's arrivals = Σ cancel_proba (model).
            day_p      = hd.loc[arr == dt, S.CANCEL_PROBA]
            expected   = float(day_p.sum())                 # risk-neutral expected freed rooms
            high_conf  = int((day_p >= threshold).sum())    # conservative: only "sure" cancels
            recs.append({"date": dt, S.HOTEL_CODE: h, "arrivals": arrivals,
                         "departures": departures, "occupancy": occupancy,
                         "expected_cancels": expected, "high_conf_cancels": high_conf})
    return pd.DataFrame.from_records(recs)


def grid_matrices(df, dates, units_by_hotel, labels=None, perf=None) -> dict:
    """Pivot `daily_grid` into one location × date matrix per metric (for the heatmap).

    Returns {'arrivals', 'departures', 'occupancy', 'expected_cancels'} -> DataFrame
    (index = location label, columns = date), all aligned to the SAME shape so the
    raster factory can annotate cells positionally. `labels` maps hotel_code ->
    display name; if given, rows are relabelled.
    """
    g = daily_grid(df, dates, units_by_hotel, perf=perf)    # long tidy grid (perf-aware)
    out = {}
    for metric in ("arrivals", "departures", "occupancy", "expected_cancels"):
        # Pivot to rows=hotel, cols=date; reindex to keep the configured hotel order.
        m = (g.pivot(index=S.HOTEL_CODE, columns="date", values=metric)
               .reindex(list(units_by_hotel)))
        if labels:                                          # optional pretty row labels
            m.index = [labels.get(h, h) for h in m.index]
        out[metric] = m
    return out


# ---------------------------------------------------------------------------
# Filtering + segment analysis + reservation-level view (for the rich page)
# ---------------------------------------------------------------------------
def apply_filters(df, *, hotels=None, channels=None, purposes=None, rooms=None,
                  risk=None, dates=None) -> pd.DataFrame:
    """Apply the sidebar filters to a scored frame. Each empty/None arg = no filter.

    Always restricts to confirmed bookings first (canceled rows are not decisions
    we still influence). Used as the GLOBAL filter; pages may narrow further.
    """
    out = confirmed(df)                                     # drop already-canceled
    if hotels:   out = out[out[S.HOTEL_CODE].isin(hotels)]      # location filter
    if channels: out = out[out[S.CHANNEL].isin(channels)]      # booking channel
    if purposes: out = out[out[S.TRAVEL_PURPOSE].isin(purposes)]  # business/leisure
    if rooms:    out = out[out[S.UNIT_GROUP].isin(rooms)]      # room category
    if risk:     out = out[out[S.RISK_BUCKET].isin(risk)]      # risk bucket
    if dates is not None:                                      # arrival within window
        out = out[out[S.ARRIVAL_DATE].isin(list(dates))]
    return out


def segment_breakdown(df, by: str) -> pd.DataFrame:
    """Per-category guest/behaviour breakdown for column `by`.

    Returns label / bookings / cancel_rate (mean proba) / expected_cancels (Σ proba),
    sorted by booking volume. Drives the arrival-segment bar charts (who are these
    guests, how do they behave) for channel / room category / travel purpose / etc.
    """
    d = confirmed(df)                                       # confirmed only
    if by not in d.columns or d.empty:                     # guard missing column/empty
        return pd.DataFrame(columns=["label", "bookings", "cancel_rate", "expected_cancels"])
    g = d.groupby(by, dropna=False)                        # one group per category value
    out = pd.DataFrame({
        "label":            [str(k) for k in g.size().index],  # category value as string
        "bookings":         g.size().values,                   # how many bookings
        "cancel_rate":      g[S.CANCEL_PROBA].mean().values,   # avg cancel probability
        "expected_cancels": g[S.CANCEL_PROBA].sum().values,    # expected # cancellations
    })
    return out.sort_values("bookings", ascending=False).reset_index(drop=True)


# Columns shown in the booking-level reservation table (in display order).
RESERVATION_COLS = [S.BOOKING_ID, S.HOTEL_CODE, S.ARRIVAL_DATE, S.UNIT_GROUP, S.CHANNEL,
                    S.TRAVEL_PURPOSE, S.LEAD_TIME_DAYS, S.LOS_NIGHTS, S.GROSS_AMOUNT,
                    S.CANCEL_PROBA, S.RISK_BUCKET]


def reservation_view(df, *, hotel_code=None, day=None) -> pd.DataFrame:
    """Booking-level table with cancel probability + risk, German headers.

    Sorted by cancel probability (highest first). Optional `hotel_code`/`day`
    narrow to a single heatmap cell (the click-drilldown). Probability is shown as
    a percentage; risk bucket is translated to German.
    """
    d = predictions_table(df, hotel_code=hotel_code, day=day)  # confirmed + filtered + sorted
    cols = [c for c in RESERVATION_COLS if c in d.columns]     # only existing columns
    out = d[cols].copy()
    # Format for display: date as ISO, probability as %, risk as German word.
    if S.ARRIVAL_DATE in out:
        out[S.ARRIVAL_DATE] = pd.to_datetime(out[S.ARRIVAL_DATE]).dt.strftime("%Y-%m-%d")
    if S.CANCEL_PROBA in out:
        out[S.CANCEL_PROBA] = (out[S.CANCEL_PROBA] * 100).round(1)   # 0..100 (%)
    if S.RISK_BUCKET in out:
        out[S.RISK_BUCKET] = out[S.RISK_BUCKET].map(S.RISK_LABELS).fillna(out[S.RISK_BUCKET])
    # Rename to the display labels (S.LABELS).
    return out.rename(columns={c: S.LABELS.get(c, c) for c in out.columns})


def recommendation_by_day(df, dates, units_by_hotel, perf=None, labels=None,
                          occ_floor: float = 0.85, threshold=S.HIGH_THR) -> pd.DataFrame:
    """Per-(Standort, Tag) overbooking recommendation that factors in BOTH the
    changing occupancy AND the day's cancellation probability.

    Rationale (why per-day, not per-window): overbooking only makes sense when a
    property is (near) full that day. So:
      * free rooms that day  = round(units * (1 - occupancy))   [occupancy from
        the performance table where available, else the booking proxy];
      * expected cancellations that day = Σ cancel_proba of that day's arrivals;
      * recommendation = 0 when occupancy < `occ_floor` (slack — no need to
        oversell); otherwise oversell the day's HIGH-CONFIDENCE cancellations
        (count of arrivals with cancel_proba >= `threshold`), capped by the
        benchmark allowance (2 for <50 units, 4 else).

    CONSERVATIVE by design (the revenue team's rule: at full capacity don't
    oversell unless we are fairly sure of a cancellation). The threshold defaults
    to the model's cost-optimal validation point and is set by the UI slider. The
    risk-neutral expectation Σ p ("Erw. Stornos") is kept as a reference column so
    analysts can see the gap between the expected and the conservative count.
    """
    g = daily_grid(df, dates, units_by_hotel, perf=perf, threshold=threshold)
    rows = []
    for r in g.itertuples():
        h = getattr(r, S.HOTEL_CODE)                        # hotel_code
        units = max(int(units_by_hotel.get(h, 0)), 1)
        occ = float(r.occupancy)                            # 0..1+
        exp = float(r.expected_cancels)                     # Σ proba (risk-neutral)
        sure = int(getattr(r, "high_conf_cancels", 0))      # count p >= threshold (conservative)
        free = max(0, round(units * (1 - occ)))             # free rooms that day (info)
        allow = 4 if units >= 50 else 2                     # benchmark allowance
        # Occupancy GATES (only oversell when near full); the conservative
        # high-confidence COUNT sets the amount, capped by the benchmark allowance.
        rec = 0 if occ < occ_floor else int(min(allow, max(0, sure)))
        rows.append({
            "Standort": (labels or {}).get(h, h),
            "Datum": pd.Timestamp(r.date).strftime("%a %d.%m."),
            "Auslastung": f"{occ*100:.0f}%",
            "Anreisen": int(r.arrivals),
            "Erw. Stornos": round(exp, 1),
            "Sichere Stornos": sure,
            "Frei": free,
            "Limit": allow,
            "Empfehlung": rec,
        })
    out = pd.DataFrame(rows)
    # Surface the strongest recommendations first.
    return out.sort_values(["Empfehlung", "Erw. Stornos"], ascending=False).reset_index(drop=True)
