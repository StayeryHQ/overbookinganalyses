"""Ableitungen (Belegung, An-/Abreisen, Zimmerkategorie, Synthese, Prognose). Reine Pandas-Funktionen."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S


def confirmed(df: pd.DataFrame) -> pd.DataFrame:
    if S.STATUS in df.columns:
        return df[df[S.STATUS] == S.STATUS_CONFIRMED]
    return df


def _overlap_counts(df, d64, group_col):
    mask = (df[S.ARRIVAL].values <= d64) & (df[S.DEPARTURE].values > d64)
    if not mask.any():
        return pd.Series(dtype=int)
    return pd.Series(df[group_col].values[mask]).value_counts()


def occupancy_long(df, dates, units_by_hotel) -> pd.DataFrame:
    df = confirmed(df)
    hotels = list(units_by_hotel)
    recs = []
    for d in dates:
        counts = _overlap_counts(df, np.datetime64(d), S.HOTEL_CODE)
        for h in hotels:
            cap = int(units_by_hotel.get(h, 0)) or 1
            occ_units = int(counts.get(h, 0))
            recs.append({"date": d, S.HOTEL_CODE: h, "occupied": occ_units,
                         "capacity": cap, "occupancy": occ_units / cap})
    return pd.DataFrame.from_records(recs)


def occupancy_matrix(df, dates, units_by_hotel, value="occupancy") -> pd.DataFrame:
    long = occupancy_long(df, dates, units_by_hotel)
    return long.pivot(index=S.HOTEL_CODE, columns="date", values=value).reindex(list(units_by_hotel))


def _count_matrix(df, dates, hotels, date_col) -> pd.DataFrame:
    df = confirmed(df)
    sub = df[df[date_col].isin(dates)]
    g = sub.groupby([S.HOTEL_CODE, date_col]).size().rename("n").reset_index()
    mat = g.pivot(index=S.HOTEL_CODE, columns=date_col, values="n")
    return mat.reindex(index=list(hotels), columns=list(dates)).fillna(0)


def arrivals_matrix(df, dates, hotels) -> pd.DataFrame:
    return _count_matrix(df, dates, hotels, S.ARRIVAL_DATE)


def departures_matrix(df, dates, hotels) -> pd.DataFrame:
    return _count_matrix(df, dates, hotels, S.DEPARTURE_DATE)


def category_occupancy_matrix(df, dates, hotel_code, caps) -> pd.DataFrame:
    df = confirmed(df)
    df = df[df[S.HOTEL_CODE] == hotel_code]
    recs = []
    for d in dates:
        counts = _overlap_counts(df, np.datetime64(d), S.UNIT_GROUP)
        for cat in S.ROOM_CATEGORIES:
            cap = int(caps.get(cat, 0)) or 1
            occ = int(counts.get(cat, 0))
            recs.append({"date": d, "category": cat, "occupancy": occ / cap, "occupied": occ, "capacity": cap})
    long = pd.DataFrame.from_records(recs)
    return long.pivot(index="category", columns="date", values="occupancy").reindex(list(S.ROOM_CATEGORIES))


def high_occupancy_days(df, dates, units_by_hotel, threshold=0.90) -> pd.DataFrame:
    long = occupancy_long(df, dates, units_by_hotel)
    hot = long[long["occupancy"] > threshold].copy()
    return hot.sort_values(["date", "occupancy"], ascending=[True, False])


def synthesis_by_day(df, dates, units_by_hotel, threshold=0.90, proba_thr=S.HIGH_THR) -> pd.DataFrame:
    df_c = confirmed(df)
    long = occupancy_long(df, dates, units_by_hotel)
    rows = []
    for d in dates:
        day = long[long["date"] == d]
        hot = day[day["occupancy"] > threshold]
        if hot.empty:
            continue
        codes = list(hot[S.HOTEL_CODE])
        arr_day = df_c[(df_c[S.ARRIVAL_DATE] == d) & (df_c[S.HOTEL_CODE].isin(codes))]
        rows.append({
            "Datum": d,
            "Standorte": len(codes),
            "Wo": ", ".join(codes),
            "Ø Belegung": round(float(hot["occupancy"].mean()), 3),
            "Erwartete Stornos": round(float(arr_day[S.CANCEL_PROBA].sum()), 1),
            "High-Risk": int((arr_day[S.CANCEL_PROBA] >= proba_thr).sum()),
        })
    return pd.DataFrame(rows)


def arrivals(df, hotel_code=None, day=None, include_canceled=False) -> pd.DataFrame:
    out = df if include_canceled else confirmed(df)
    if hotel_code:
        out = out[out[S.HOTEL_CODE] == hotel_code]
    if day is not None:
        out = out[out[S.ARRIVAL_DATE] == pd.Timestamp(day).normalize()]
    return out.copy()


def arrivals_kpis(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return dict(n=0, cancelable=0, refundable=0, non_refundable=0, long_stay=0,
                    corporate=0, group=0, international=0, avg_lead_time=0.0, avg_los=0.0,
                    revenue=0.0, avg_cancel_proba=0.0, likely_cancels=0)
    return dict(
        n=n,
        cancelable=int(sub[S.IS_CANCELABLE].sum()),
        refundable=int(sub[S.IS_REFUNDABLE].sum()),
        non_refundable=int((~sub[S.IS_REFUNDABLE]).sum()),
        long_stay=int((sub[S.STAY_BUCKET] == "long").sum()),
        corporate=int(sub[S.IS_CORPORATE].sum()),
        group=int(sub[S.IS_GROUP].sum()),
        international=int(sub[S.IS_INTERNATIONAL].sum()),
        avg_lead_time=float(sub[S.LEAD_TIME_DAYS].mean()),
        avg_los=float(sub[S.LOS_NIGHTS].mean()),
        revenue=float(sub[S.GROSS_AMOUNT].sum()),
        avg_cancel_proba=float(sub[S.CANCEL_PROBA].mean()),
        likely_cancels=int((sub[S.CANCEL_PROBA] >= S.HIGH_THR).sum()),
    )


def breakdown_by(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame(columns=[col, "Anzahl", "Anteil", "Ø Storno-Wkt.", "Umsatz (€)"])
    g = sub.groupby(col, dropna=False)
    out = pd.DataFrame({
        "Anzahl": g.size(),
        "Ø Storno-Wkt.": g[S.CANCEL_PROBA].mean().round(3),
        "Umsatz (€)": g[S.GROSS_AMOUNT].sum().round(0),
    }).reset_index()
    out["Anteil"] = (out["Anzahl"] / out["Anzahl"].sum()).round(3)
    return out.sort_values("Anzahl", ascending=False)[[col, "Anzahl", "Anteil", "Ø Storno-Wkt.", "Umsatz (€)"]]


def prediction_long(df, dates, units_by_hotel, threshold=S.HIGH_THR) -> pd.DataFrame:
    df = confirmed(df)
    win = df[df[S.ARRIVAL_DATE].isin(dates)]
    grouped = {k: g for k, g in win.groupby([S.HOTEL_CODE, S.ARRIVAL_DATE])}
    recs = []
    for h in units_by_hotel:
        for d in dates:
            g = grouped.get((h, d))
            if g is None or g.empty:
                recs.append({"date": d, S.HOTEL_CODE: h, "arrivals": 0,
                             "expected_cancels": 0.0, "high_risk": 0})
            else:
                recs.append({"date": d, S.HOTEL_CODE: h, "arrivals": len(g),
                             "expected_cancels": float(g[S.CANCEL_PROBA].sum()),
                             "high_risk": int((g[S.CANCEL_PROBA] >= threshold).sum())})
    return pd.DataFrame.from_records(recs)


def prediction_matrix(df, dates, units_by_hotel, threshold=S.HIGH_THR, value="expected_cancels") -> pd.DataFrame:
    long = prediction_long(df, dates, units_by_hotel, threshold)
    return long.pivot(index=S.HOTEL_CODE, columns="date", values=value).reindex(list(units_by_hotel))


def recommendation_table(df, dates, units_by_hotel, threshold, labels) -> pd.DataFrame:
    pl = prediction_long(df, dates, units_by_hotel, threshold)
    horizon = max(len(dates), 1)
    rows = []
    for h in units_by_hotel:
        g = pl[pl[S.HOTEL_CODE] == h]
        expected = float(g["expected_cancels"].sum())
        allow = 4 if units_by_hotel[h] >= 50 else 2
        per_day = expected / horizon
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
    return pd.DataFrame(rows).sort_values("Empfehlung", ascending=False)


def predictions_table(df, dates=None, hotel_code=None, day=None) -> pd.DataFrame:
    out = confirmed(df)
    if dates is not None:
        out = out[out[S.ARRIVAL_DATE].isin(dates)]
    if hotel_code:
        out = out[out[S.HOTEL_CODE] == hotel_code]
    if day is not None:
        out = out[out[S.ARRIVAL_DATE] == pd.Timestamp(day).normalize()]
    return out.sort_values(S.CANCEL_PROBA, ascending=False).copy()
