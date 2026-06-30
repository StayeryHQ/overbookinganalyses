# dash_app/backend/real.py
# ---------------------------------------------------------------------------
# Real backend: a thin adapter over the project's `src.scoring`. PORTED from
# streamlit_app/backend/real.py. It calls src.score_upcoming() (BigQuery ->
# features -> model) and maps the result onto the canonical schema, so the
# pages see a single canonical schema.
#
# IMPORTANT: `src` is imported LAZILY (inside the function) so importing the UI
# never depends on a trained model / joblib / xgboost being present. With no
# model on disk this raises a clear RuntimeError; the facade then falls back to
# import. NEVER call BigQuery directly — only through src.score_upcoming.
#
# Model selection: `model_name` flows straight into src.score_upcoming(), so the
# model dropdown in the UI selects the model with zero page changes.
# ---------------------------------------------------------------------------

from __future__ import annotations

# `sys` is used to put the repo root on the path so `import src` resolves.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical schema constants.
from . import schema as S

# Repo root = three levels up (real.py -> backend/ -> dash_app/ -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_src():
    """Import the `src` package lazily, with a clear error if it can't load."""
    # Make sure the repo root is importable so `import src` finds the package.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import src  # noqa: F401  (imported for side effect of availability)
        return src
    except Exception as e:  # noqa: BLE001 — surface ANY import failure clearly
        raise RuntimeError(
            "Real-Backend nicht verfügbar: konnte `src` nicht importieren "
            f"({type(e).__name__}: {e}). Stelle sicher, dass Modelle trainiert "
            "sind (Notebooks 01-03) und die Roster-Artefakte vorhanden sind."
        ) from e


def _city_lookup() -> dict[str, str]:
    """Map hotel_code -> city from configs/locations.yaml."""
    from .locations import _load_locations
    return {hc: city for hc, city, _ in _load_locations()}


def get_scored_bookings(model_name: str | None = None, horizon_days: int = 35,
                        force_refresh: bool = False) -> pd.DataFrame:
    """Score upcoming bookings with the real model, mapped to the canonical schema.

    Parameters
    ----------
    model_name : str | None
        Registry name passed straight to src.score_upcoming (None = auto-pick
        via best_model(): best AP among well-calibrated models). Swap-model lever.
    horizon_days : int
        Accepted for signature parity (src decides its own horizon).
    force_refresh : bool
        Re-pull from BigQuery instead of using the parquet cache.
    """
    # Lazily get the src package (raises a clear error if unavailable).
    src = _import_src()
    # Score upcoming arrivals. `model_name` is the swap point; save=True writes
    # the parquet cache as the Streamlit backend does.
    feat = src.score_upcoming(model_name=model_name, force_refresh=force_refresh, save=True)
    # No rows -> return an empty frame with the right columns so pages don't crash.
    if feat is None or len(feat) == 0:
        return pd.DataFrame(columns=list(S.COLUMNS))

    # HAZARD = primary engine: if a trained hazard model is on disk, override
    # cancel_proba with the horizon-aware survival-product P(cancel before arrival)
    # (the decision is made d=1..14 days out). The static model from
    # score_upcoming is the fallback when no hazard model exists or it errors.
    try:
        from src import hazard as _HZ
        if _HZ.hazard_available():
            _hz = _HZ.load_hazard()
            _b = feat.copy()
            if "lead" not in _b.columns and "lead_time_days" in _b.columns:
                _b["lead"] = _b["lead_time_days"]
            feat = feat.copy()
            feat["cancel_proba"] = _HZ.score_upcoming_hazard(_hz, _b)
            feat["model_used"] = "hazard"
    except Exception as e:  # noqa: BLE001 — fall back to the static score
        print(f"real: hazard serving skipped, using static score ({type(e).__name__}: {e})")

    # Work on a copy and coerce the raw src columns into the canonical shape.
    df = feat.copy()
    today = pd.Timestamp.today().normalize()
    arrival = pd.to_datetime(df.get("arrival"), errors="coerce")
    departure = pd.to_datetime(df.get("departure"), errors="coerce")
    created = pd.to_datetime(df.get("created"), errors="coerce")

    # Cancellation fee + refundability derived from the fee amount.
    canc_fee = pd.to_numeric(df.get("cancellationFee_fee_amount"), errors="coerce").fillna(0.0)
    rate_cat_raw = (df.get("ratePlan_category").astype("string")
                    if "ratePlan_category" in df else pd.Series("", index=df.index))
    is_ref = canc_fee.eq(0.0)
    # Days until arrival relative to today.
    days_until = ((arrival.dt.normalize() - today) / pd.Timedelta(days=1))

    # Company / group names, coerced to plain strings.
    company = df.get("company_name")
    company = company.astype("string").fillna("") if company is not None else pd.Series("", index=df.index)
    group = df.get("groupName")
    group = group.astype("string").fillna("") if group is not None else pd.Series("", index=df.index)
    code = _city_lookup()

    # Build the canonical frame column-by-column (mirrors the Streamlit adapter).
    out = pd.DataFrame({
        S.HOTEL_CODE: df.get("property_code"),
        S.CITY: df.get("property_code").map(code) if "property_code" in df else "",
        S.PROPERTY_NAME: df.get("property_name"),
        S.STATUS: S.STATUS_CONFIRMED,   # upcoming arrivals are confirmed by definition
        S.ARRIVAL: arrival,
        S.DEPARTURE: departure,
        S.CREATED: created,
        S.ARRIVAL_DATE: arrival.dt.normalize(),
        S.DEPARTURE_DATE: departure.dt.normalize(),
        S.DAYS_UNTIL_ARRIVAL: days_until.astype("Int64"),
        S.LEAD_TIME_DAYS: pd.to_numeric(df.get("lead_time_days"), errors="coerce"),
        S.LOS_NIGHTS: pd.to_numeric(df.get("los_nights"), errors="coerce"),
        S.STAY_BUCKET: df.get("stay_bucket"),
        S.ADULTS: pd.to_numeric(df.get("adults_n"), errors="coerce").astype("Int64"),
        S.UNIT_GROUP: df.get("unitGroup_name"),
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

    # Stable booking ids.
    out[S.BOOKING_ID] = [f"{hc}-{i:05d}"
                         for i, hc in enumerate(out[S.HOTEL_CODE].astype("string").fillna("NA"), start=1)]
    # Fallback risk bucket if src didn't supply one.
    if out[S.RISK_BUCKET].isna().all():
        p = out[S.CANCEL_PROBA].fillna(0.0)
        out[S.RISK_BUCKET] = np.where(p >= S.HIGH_THR, "high",
                              np.where(p >= S.LOW_THR, "uncertain", "low"))
    # Return the canonical columns in order.
    return out[list(S.COLUMNS)].reset_index(drop=True)
