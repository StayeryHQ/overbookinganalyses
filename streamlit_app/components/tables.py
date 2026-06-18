"""Schema-bewusste Buchungs-Tabelle mit gestylten Spalten.

Baut aus dem kanonischen Backend-Schema einen Anzeige-DataFrame + passende
``st.column_config`` (deutsche Labels, Fortschrittsbalken für die
Storno-Wahrscheinlichkeit, Häkchen für Booleans, €-Formate).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import schema as S

_RISK_DE = {"high": "hoch", "uncertain": "unsicher", "low": "niedrig"}

_DEFAULT_COLS = [
    S.BOOKING_ID, S.HOTEL_CODE, S.ARRIVAL, S.LOS_NIGHTS, S.RATE_CATEGORY,
    S.IS_REFUNDABLE, S.IS_CANCELABLE, S.LEAD_TIME_DAYS, S.CHANNEL,
    S.TRAVEL_PURPOSE, S.IS_CORPORATE, S.COMPANY_NAME, S.IS_GROUP,
    S.GROSS_AMOUNT, S.CANCEL_PROBA, S.RISK_BUCKET,
]


def booking_view(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """(Anzeige-DataFrame, column_config) für ``st.dataframe`` zurückgeben."""
    cols = [c for c in (columns or _DEFAULT_COLS) if c in df.columns]
    disp = df[cols].copy()

    # Storno-Wahrscheinlichkeit als Prozent (für Fortschrittsbalken 0..100).
    if S.CANCEL_PROBA in disp:
        disp[S.CANCEL_PROBA] = (pd.to_numeric(disp[S.CANCEL_PROBA], errors="coerce") * 100).round(0)
    if S.RISK_BUCKET in disp:
        disp[S.RISK_BUCKET] = disp[S.RISK_BUCKET].map(_RISK_DE).fillna(disp[S.RISK_BUCKET])

    cc: dict = {}
    L = S.LABELS_DE
    if S.ARRIVAL in disp:
        cc[S.ARRIVAL] = st.column_config.DatetimeColumn(L[S.ARRIVAL], format="DD.MM.YYYY")
    if S.LOS_NIGHTS in disp:
        cc[S.LOS_NIGHTS] = st.column_config.NumberColumn(L[S.LOS_NIGHTS], format="%d")
    if S.LEAD_TIME_DAYS in disp:
        cc[S.LEAD_TIME_DAYS] = st.column_config.NumberColumn(L[S.LEAD_TIME_DAYS], format="%d")
    if S.GROSS_AMOUNT in disp:
        cc[S.GROSS_AMOUNT] = st.column_config.NumberColumn(L[S.GROSS_AMOUNT], format="%.0f €")
    if S.CANCELLATION_FEE in disp:
        cc[S.CANCELLATION_FEE] = st.column_config.NumberColumn(L[S.CANCELLATION_FEE], format="%.0f €")
    if S.CANCEL_PROBA in disp:
        cc[S.CANCEL_PROBA] = st.column_config.ProgressColumn(
            L[S.CANCEL_PROBA], format="%d%%", min_value=0, max_value=100)
    for b in (S.IS_REFUNDABLE, S.IS_CANCELABLE, S.IS_CORPORATE, S.IS_GROUP, S.IS_INTERNATIONAL,
              S.HAS_PROMO, S.HAS_CORPORATE_CODE):
        if b in disp:
            cc[b] = st.column_config.CheckboxColumn(L[b])
    for t in (S.BOOKING_ID, S.HOTEL_CODE, S.CITY, S.PROPERTY_NAME, S.RATE_CATEGORY,
              S.RATE_PLAN, S.CHANNEL, S.TRAVEL_PURPOSE, S.GUARANTEE_TYPE, S.COUNTRY_CODE,
              S.COMPANY_NAME, S.GROUP_NAME, S.RISK_BUCKET, S.STAY_BUCKET):
        if t in disp:
            cc[t] = st.column_config.TextColumn(L.get(t, t))
    return disp, cc
