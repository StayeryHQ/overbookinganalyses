"""Synthetischer, deterministischer Buchungs-Snapshot im kanonischen Schema."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import schema as S

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCATIONS_YAML = _REPO_ROOT / "configs" / "locations.yaml"

_CITY_BASE_PRICE = {
    "Frankfurt": 132, "Berlin": 128, "Köln": 118, "Bremen": 104,
    "Bielefeld": 96, "Bochum": 94, "Osnabrück": 98, "Gütersloh": 92,
    "Wolfsburg": 110,
}
_DEFAULT_PRICE = 100

_CHANNELS = np.array(["Direct", "Booking.com", "Expedia", "Corporate", "HRS"])
_CHANNEL_P = np.array([0.40, 0.25, 0.12, 0.15, 0.08])
_OTA = ["Booking.com", "Expedia", "HRS"]

_COMPANIES = np.array([
    "Volkswagen AG", "SAP SE", "Siemens AG", "Robert Bosch GmbH",
    "Deutsche Bahn AG", "Lufthansa Group", "BASF SE", "Bayer AG",
    "Continental AG", "Festo SE", "Miele & Cie. KG", "Claas KGaA",
])
_COUNTRIES = np.array(["DE", "DE", "DE", "DE", "DE", "AT", "CH", "NL", "FR", "GB", "US"])
_GUARANTEES = np.array(["CreditCard", "Prepayment", "Company", "None"])
_ROOMS = np.array(S.ROOM_CATEGORIES)
_ROOM_P = np.array([S.ROOM_CATEGORY_SPLIT[c] for c in S.ROOM_CATEGORIES])


@lru_cache(maxsize=8)
def _load_locations() -> tuple[tuple[str, str, int], ...]:
    with _LOCATIONS_YAML.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    out = []
    for loc in data.get("locations", []):
        units = int(loc.get("units_total") or 0) or 60
        out.append((loc["hotel_code"], loc.get("city", ""), units))
    return tuple(out)


def _gen_location(rng, hotel_code, city, units, today, horizon):
    occ_target = rng.uniform(0.60, 0.90)
    mean_los = 4.65
    window = horizon + 4
    n = max(int(occ_target * units * window / mean_los), 40)
    n = int(n * 1.12)

    arr_off = rng.randint(-3, horizon + 1, size=n)
    arrival = today + pd.to_timedelta(arr_off, unit="D")

    u = rng.rand(n)
    los = np.where(u < 0.60, rng.randint(1, 4, n),
          np.where(u < 0.90, rng.randint(4, 8, n), rng.randint(8, 29, n))).astype(int)
    departure = arrival + pd.to_timedelta(los, unit="D")

    lead = np.clip(rng.gamma(2.0, 14.0, n).astype(int), 0, 200)
    created = arrival - pd.to_timedelta(lead, unit="D")

    cat = rng.choice(S.RATE_CATEGORIES, size=n, p=[0.45, 0.20, 0.35])
    is_ref = cat != "Non-Ref"
    free_window = np.where(cat == "Flex", 1, np.where(cat == "Semi-Flex", 3, 0))
    is_cancelable = is_ref & (arr_off > free_window)
    rate_plan = np.where(cat == "Flex", "Flexible Rate",
                np.where(cat == "Semi-Flex", "Semi-Flex Rate", "Non-Refundable"))

    base = _CITY_BASE_PRICE.get(city, _DEFAULT_PRICE)
    per_night = np.round(base * rng.uniform(0.78, 1.45, n), 2)
    gross = np.round(per_night * los, 2)
    canc_fee = np.where(cat == "Non-Ref", gross,
               np.where(cat == "Semi-Flex", per_night, 0.0)).round(2)

    channel = rng.choice(_CHANNELS, size=n, p=_CHANNEL_P)
    is_ota = np.isin(channel, _OTA)
    is_corp = (channel == "Corporate") | (rng.rand(n) < 0.12)
    purpose = np.where(is_corp, "Business",
                       rng.choice(["Business", "Leisure"], size=n, p=[0.40, 0.60]))
    company = np.where(is_corp, rng.choice(_COMPANIES, size=n), "")
    has_corp_code = is_corp & (rng.rand(n) < 0.85)
    is_group = rng.rand(n) < 0.06
    group_name = np.where(is_group,
                          np.char.add("Gruppe ", rng.randint(100, 999, n).astype(str)), "")
    country = rng.choice(_COUNTRIES, size=n)
    is_intl = country != "DE"
    has_promo = (~is_corp) & (rng.rand(n) < 0.15)
    adults = rng.choice([1, 2, 2, 2, 3], size=n)
    guarantee = np.where(is_corp, "Company", rng.choice(_GUARANTEES, size=n))
    unit_group = rng.choice(_ROOMS, size=n, p=_ROOM_P)

    z = (
        -1.75
        + 0.011 * lead
        + 0.55 * is_ota.astype(float)
        + 0.45 * (cat == "Flex").astype(float)
        - 0.55 * (cat == "Non-Ref").astype(float)
        - 0.35 * (purpose == "Business").astype(float)
        + 0.25 * is_group.astype(float)
        - 0.035 * los
        + 0.25 * is_intl.astype(float)
        - 0.20 * is_corp.astype(float)
    )
    z = z + rng.normal(0, 0.85, n)
    proba = np.clip(1.0 / (1.0 + np.exp(-z)), 0.01, 0.98).round(4)

    status = np.where(rng.rand(n) < 0.12, S.STATUS_CANCELED, S.STATUS_CONFIRMED)

    return pd.DataFrame({
        S.HOTEL_CODE: hotel_code,
        S.CITY: city,
        S.PROPERTY_NAME: f"STAYERY {city}",
        S.STATUS: status,
        S.ARRIVAL: arrival,
        S.DEPARTURE: departure,
        S.CREATED: created,
        S.DAYS_UNTIL_ARRIVAL: arr_off.astype(int),
        S.LEAD_TIME_DAYS: lead,
        S.LOS_NIGHTS: los,
        S.ADULTS: adults,
        S.UNIT_GROUP: unit_group,
        S.RATE_PLAN: rate_plan,
        S.RATE_CATEGORY: cat,
        S.IS_REFUNDABLE: is_ref,
        S.IS_CANCELABLE: is_cancelable,
        S.CANCELLATION_FEE: canc_fee,
        S.CHANNEL: channel,
        S.TRAVEL_PURPOSE: purpose,
        S.GUARANTEE_TYPE: guarantee,
        S.COUNTRY_CODE: country,
        S.IS_INTERNATIONAL: is_intl,
        S.HAS_PROMO: has_promo,
        S.HAS_CORPORATE_CODE: has_corp_code,
        S.IS_CORPORATE: is_corp,
        S.IS_GROUP: is_group,
        S.COMPANY_NAME: company,
        S.GROUP_NAME: group_name,
        S.GROSS_AMOUNT: gross,
        S.GROSS_PER_NIGHT: per_night,
        S.CANCEL_PROBA: proba,
    })


@lru_cache(maxsize=8)
def generate(seed: int = 42, horizon_days: int = 35) -> pd.DataFrame:
    rng = np.random.RandomState(int(seed))
    today = pd.Timestamp.today().normalize()
    parts = [_gen_location(rng, hc, city, units, today, horizon_days)
             for hc, city, units in _load_locations()]
    df = pd.concat(parts, ignore_index=True)

    df[S.ARRIVAL_DATE] = df[S.ARRIVAL].dt.normalize()
    df[S.DEPARTURE_DATE] = df[S.DEPARTURE].dt.normalize()
    df[S.STAY_BUCKET] = pd.cut(df[S.LOS_NIGHTS], bins=[-1, 2, 6, 365],
                               labels=["short", "mid", "long"]).astype("object")
    df[S.RISK_BUCKET] = np.where(
        df[S.CANCEL_PROBA] >= S.HIGH_THR, "high",
        np.where(df[S.CANCEL_PROBA] >= S.LOW_THR, "uncertain", "low"))
    df[S.BOOKING_ID] = [f"{hc}-{i:05d}" for i, hc in enumerate(df[S.HOTEL_CODE], start=1)]
    return df[list(S.COLUMNS)].reset_index(drop=True)
