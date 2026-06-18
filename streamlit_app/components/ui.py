"""Gemeinsame Sidebar-Filter + Erklär-Tooltips."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import schema as S


def explain(title: str, body: str) -> None:
    try:
        with st.popover(title):
            st.markdown(body)
    except Exception:
        with st.expander(title):
            st.markdown(body)


def sidebar_filters(bookings, units, labels, *, prefix: str,
                    want_window=True, want_threshold=False, want_rooms=True) -> dict:
    sb = st.sidebar
    sb.markdown("## Filter")
    out: dict = {}

    codes = sb.multiselect("Standorte", list(units), default=list(units),
                           format_func=lambda c: labels.get(c, c), key=f"{prefix}_codes")
    out["codes"] = codes or list(units)

    if want_window:
        out["horizon"] = sb.selectbox("Zeitfenster", [7, 14, 21], index=1,
                                      format_func=lambda d: f"{d} Tage", key=f"{prefix}_hz")
    if want_threshold:
        out["threshold"] = sb.slider("High-Risk-Schwelle", 0.50, 0.90, value=float(S.HIGH_THR),
                                     step=0.05, key=f"{prefix}_thr",
                                     help="Ab dieser Storno-Wahrscheinlichkeit gilt eine "
                                     "Einzelbuchung als High-Risk (manuelle Prüfung). "
                                     "Die Overbooking-Empfehlung nutzt den Erwartungswert, "
                                     "nicht diese Schwelle.")
    if want_rooms:
        out["rooms"] = sb.multiselect("Zimmerkategorie", list(S.ROOM_CATEGORIES),
                                      default=list(S.ROOM_CATEGORIES), key=f"{prefix}_rooms")

    out["channels"] = sb.multiselect("Kanal", sorted(bookings[S.CHANNEL].dropna().unique()),
                                     default=[], key=f"{prefix}_ch", help="Leer = alle.")
    out["segments"] = sb.multiselect("Segment", ["Firmenkunden", "Gruppen", "Privat"],
                                     default=[], key=f"{prefix}_seg", help="Leer = alle.")
    comps = sorted(c for c in bookings[S.COMPANY_NAME].dropna().unique() if c)
    out["companies"] = sb.multiselect("Firma", comps, default=[], key=f"{prefix}_co",
                                      help="Leer = alle.")
    out["risk"] = sb.multiselect("Risiko", list(S.RISK_BUCKETS), default=[],
                                 format_func=lambda r: S.RISK_LABELS_DE[r], key=f"{prefix}_risk",
                                 help="Leer = alle.")
    out["include_canceled"] = sb.toggle("Stornierte einbeziehen", value=False,
                                        key=f"{prefix}_canc",
                                        help="Standard aus: bereits stornierte Buchungen "
                                        "sind ausgeschlossen.")
    return out


def base_frame(bookings, f) -> pd.DataFrame:
    out = bookings
    if not f.get("include_canceled"):
        out = out[out[S.STATUS] == S.STATUS_CONFIRMED]
    if f.get("codes"):
        out = out[out[S.HOTEL_CODE].isin(f["codes"])]
    return out.copy()


def apply_filters(bookings, f) -> pd.DataFrame:
    out = base_frame(bookings, f)
    if f.get("rooms") and len(f["rooms"]) < len(S.ROOM_CATEGORIES):
        out = out[out[S.UNIT_GROUP].isin(f["rooms"])]
    if f.get("channels"):
        out = out[out[S.CHANNEL].isin(f["channels"])]
    if f.get("companies"):
        out = out[out[S.COMPANY_NAME].isin(f["companies"])]
    if f.get("risk"):
        out = out[out[S.RISK_BUCKET].isin(f["risk"])]
    seg = f.get("segments")
    if seg:
        m = pd.Series(False, index=out.index)
        if "Firmenkunden" in seg:
            m |= out[S.IS_CORPORATE]
        if "Gruppen" in seg:
            m |= out[S.IS_GROUP]
        if "Privat" in seg:
            m |= ~(out[S.IS_CORPORATE] | out[S.IS_GROUP])
        out = out[m]
    return out


def canceled_count(bookings, f) -> int:
    sub = bookings
    if f.get("codes"):
        sub = sub[sub[S.HOTEL_CODE].isin(f["codes"])]
    return int((sub[S.STATUS] == S.STATUS_CANCELED).sum())
