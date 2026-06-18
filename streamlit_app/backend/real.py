"""Real-Backend: dünner Adapter auf das echte ``src``-Scoring.

Wird aktiv, sobald das Modell + die erwartete Spalte stehen. Ruft
``src.score_upcoming()`` (BigQuery → Features → Modell) und mappt dessen
Spalten auf das **kanonische Schema** aus ``schema.py``. Damit sehen die Pages
keinen Unterschied zum Dummy.

WICHTIG: Dieser Adapter importiert ``src`` bewusst **lazy** (erst im
Funktionsaufruf), damit der Import der UI nicht an matplotlib/joblib/xgboost
oder fehlenden Modellen hängt. Solange kein trainiertes Modell auf Disk liegt,
wirft der Aufruf eine klare Meldung — die UI fällt dann auf den Dummy zurück
bzw. zeigt den Hinweis an.

→ Umschalten auf echt:  Umgebungsvariable ``OVERBOOKING_BACKEND=real`` setzen
   ODER in der Seite „Datenaktualisierung" den Modus-Schalter umlegen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_src():
    """``src``-Package lazy importieren (mit Pfad-Setup, klare Fehlermeldung)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import src  # noqa: F401
        return src
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Real-Backend nicht verfügbar: konnte `src` nicht importieren "
            f"({type(e).__name__}: {e}). Stelle sicher, dass Modelle trainiert "
            "sind (Notebooks 01–04) und die erwartete Spalte vorhanden ist."
        ) from e


def _city_lookup() -> dict[str, str]:
    from .dummy import _load_locations
    return {hc: city for hc, city, _ in _load_locations()}


def get_scored_bookings(horizon_days: int = 35, force_refresh: bool = False) -> pd.DataFrame:
    """Echte, bewertete Upcoming-Buchungen → kanonisches Schema."""
    src = _import_src()
    feat = src.score_upcoming(model_name=None, force_refresh=force_refresh, save=True)
    if feat is None or len(feat) == 0:
        return pd.DataFrame(columns=list(S.COLUMNS))

    df = feat.copy()
    today = pd.Timestamp.today().normalize()
    arrival = pd.to_datetime(df.get("arrival"), errors="coerce")
    departure = pd.to_datetime(df.get("departure"), errors="coerce")
    created = pd.to_datetime(df.get("created"), errors="coerce")

    canc_fee = pd.to_numeric(df.get("cancellationFee_fee_amount"), errors="coerce").fillna(0.0)
    rate_cat_raw = df.get("ratePlan_category").astype("string") if "ratePlan_category" in df else pd.Series("", index=df.index)
    is_ref = canc_fee.eq(0.0)
    days_until = ((arrival.dt.normalize() - today) / pd.Timedelta(days=1))

    company = df.get("company_name")
    company = company.astype("string").fillna("") if company is not None else pd.Series("", index=df.index)
    group = df.get("groupName")
    group = group.astype("string").fillna("") if group is not None else pd.Series("", index=df.index)
    code = _city_lookup()

    out = pd.DataFrame({
        S.HOTEL_CODE: df.get("property_code"),
        S.CITY: df.get("property_code").map(code) if "property_code" in df else "",
        S.PROPERTY_NAME: df.get("property_name"),
        S.ARRIVAL: arrival,
        S.DEPARTURE: departure,
        S.CREATED: created,
        S.ARRIVAL_DATE: arrival.dt.normalize(),
        S.DAYS_UNTIL_ARRIVAL: days_until.astype("Int64"),
        S.LEAD_TIME_DAYS: pd.to_numeric(df.get("lead_time_days"), errors="coerce"),
        S.LOS_NIGHTS: pd.to_numeric(df.get("los_nights"), errors="coerce"),
        S.STAY_BUCKET: df.get("stay_bucket"),
        S.ADULTS: pd.to_numeric(df.get("adults_n"), errors="coerce").astype("Int64"),
        S.RATE_PLAN: df.get("ratePlan_name"),
        S.RATE_CATEGORY: rate_cat_raw,
        S.IS_REFUNDABLE: is_ref,
        S.IS_CANCELABLE: is_ref & (days_until > 1),
        S.CANCELLATION_FEE: canc_fee.round(2),
        S.CHANNEL: df.get("channelCode"),
        S.TRAVEL_PURPOSE: df.get("travelPurpose"),
        S.GUARANTEE_TYPE: df.get("guaranteeType"),
        S.COUNTRY_CODE: df.get("primaryGuest_address_countryCode"),
        S.IS_INTERNATIONAL: pd.to_numeric(df.get("is_international"), errors="coerce").fillna(0).astype(bool),
        S.HAS_PROMO: pd.to_numeric(df.get("has_promo"), errors="coerce").fillna(0).astype(bool),
        S.HAS_CORPORATE_CODE: pd.to_numeric(df.get("has_corporate_code"), errors="coerce").fillna(0).astype(bool),
        S.IS_CORPORATE: (company.str.len() > 0) | pd.to_numeric(df.get("has_corporate_code"), errors="coerce").fillna(0).astype(bool),
        S.IS_GROUP: pd.to_numeric(df.get("has_group"), errors="coerce").fillna(0).astype(bool),
        S.COMPANY_NAME: company,
        S.GROUP_NAME: group,
        S.GROSS_AMOUNT: pd.to_numeric(df.get("gross_amount"), errors="coerce"),
        S.GROSS_PER_NIGHT: pd.to_numeric(df.get("gross_per_night"), errors="coerce"),
        S.CANCEL_PROBA: pd.to_numeric(df.get("cancel_proba"), errors="coerce"),
        S.RISK_BUCKET: df.get("risk_bucket"),
    })

    out[S.BOOKING_ID] = [f"{hc}-{i:05d}" for i, hc in enumerate(out[S.HOTEL_CODE].astype("string").fillna("NA"), start=1)]
    # Fallback-Risk falls src es nicht mitliefert.
    if out[S.RISK_BUCKET].isna().all():
        p = out[S.CANCEL_PROBA].fillna(0.0)
        out[S.RISK_BUCKET] = np.where(p >= S.HIGH_THR, "high",
                              np.where(p >= S.LOW_THR, "uncertain", "low"))
    return out[list(S.COLUMNS)].reset_index(drop=True)
