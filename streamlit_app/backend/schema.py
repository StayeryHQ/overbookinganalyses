"""Kanonisches Schema für bewertete Buchungen. Beide Backends liefern genau diese Spalten."""

from __future__ import annotations

from typing import Final

LOW_THR: Final[float] = 0.60
HIGH_THR: Final[float] = 0.75

RISK_BUCKETS: Final[tuple[str, ...]] = ("low", "uncertain", "high")
RISK_LABELS_DE: Final[dict[str, str]] = {"low": "niedrig", "uncertain": "unsicher", "high": "hoch"}

RATE_CATEGORIES: Final[tuple[str, ...]] = ("Flex", "Semi-Flex", "Non-Ref")
ROOM_CATEGORIES: Final[tuple[str, ...]] = ("BIG", "BIGGER", "UPPER", "UPPER AIR")
ROOM_CATEGORY_SPLIT: Final[dict[str, float]] = {
    "BIG": 0.42, "BIGGER": 0.33, "UPPER": 0.18, "UPPER AIR": 0.07,
}

STATUS_CONFIRMED: Final[str] = "Confirmed"
STATUS_CANCELED: Final[str] = "Canceled"

BOOKING_ID = "booking_id"
HOTEL_CODE = "hotel_code"
CITY = "city"
PROPERTY_NAME = "property_name"
STATUS = "status"

ARRIVAL = "arrival"
DEPARTURE = "departure"
CREATED = "created"
ARRIVAL_DATE = "arrival_date"
DEPARTURE_DATE = "departure_date"
DAYS_UNTIL_ARRIVAL = "days_until_arrival"

LEAD_TIME_DAYS = "lead_time_days"
LOS_NIGHTS = "los_nights"
STAY_BUCKET = "stay_bucket"
ADULTS = "adults"
UNIT_GROUP = "unit_group"

RATE_PLAN = "rate_plan"
RATE_CATEGORY = "rate_category"
IS_REFUNDABLE = "is_refundable"
IS_CANCELABLE = "is_cancelable"
CANCELLATION_FEE = "cancellation_fee"

CHANNEL = "channel"
TRAVEL_PURPOSE = "travel_purpose"
GUARANTEE_TYPE = "guarantee_type"
COUNTRY_CODE = "country_code"
IS_INTERNATIONAL = "is_international"

HAS_PROMO = "has_promo"
HAS_CORPORATE_CODE = "has_corporate_code"
IS_CORPORATE = "is_corporate"
IS_GROUP = "is_group"
COMPANY_NAME = "company_name"
GROUP_NAME = "group_name"

GROSS_AMOUNT = "gross_amount"
GROSS_PER_NIGHT = "gross_per_night"

CANCEL_PROBA = "cancel_proba"
RISK_BUCKET = "risk_bucket"

COLUMNS: Final[tuple[str, ...]] = (
    BOOKING_ID, HOTEL_CODE, CITY, PROPERTY_NAME, STATUS,
    ARRIVAL, DEPARTURE, CREATED, ARRIVAL_DATE, DEPARTURE_DATE, DAYS_UNTIL_ARRIVAL,
    LEAD_TIME_DAYS, LOS_NIGHTS, STAY_BUCKET, ADULTS, UNIT_GROUP,
    RATE_PLAN, RATE_CATEGORY, IS_REFUNDABLE, IS_CANCELABLE, CANCELLATION_FEE,
    CHANNEL, TRAVEL_PURPOSE, GUARANTEE_TYPE, COUNTRY_CODE, IS_INTERNATIONAL,
    HAS_PROMO, HAS_CORPORATE_CODE, IS_CORPORATE, IS_GROUP, COMPANY_NAME, GROUP_NAME,
    GROSS_AMOUNT, GROSS_PER_NIGHT, CANCEL_PROBA, RISK_BUCKET,
)

LABELS_DE: Final[dict[str, str]] = {
    BOOKING_ID: "Buchung",
    HOTEL_CODE: "Standort",
    CITY: "Stadt",
    PROPERTY_NAME: "Hotel",
    STATUS: "Status",
    ARRIVAL: "Anreise",
    DEPARTURE: "Abreise",
    ARRIVAL_DATE: "Anreise",
    DEPARTURE_DATE: "Abreise",
    DAYS_UNTIL_ARRIVAL: "Tage bis Anreise",
    LEAD_TIME_DAYS: "Lead-Time (Tage)",
    LOS_NIGHTS: "Nächte",
    STAY_BUCKET: "Aufenthalt",
    ADULTS: "Personen",
    UNIT_GROUP: "Zimmerkategorie",
    RATE_PLAN: "Ratenplan",
    RATE_CATEGORY: "Raten-Kategorie",
    IS_REFUNDABLE: "Erstattbar",
    IS_CANCELABLE: "Noch stornierbar",
    CANCELLATION_FEE: "Storno-Gebühr (€)",
    CHANNEL: "Kanal",
    TRAVEL_PURPOSE: "Reisezweck",
    GUARANTEE_TYPE: "Garantie",
    COUNTRY_CODE: "Land",
    IS_INTERNATIONAL: "International",
    HAS_PROMO: "Promo",
    HAS_CORPORATE_CODE: "Firmencode",
    IS_CORPORATE: "Firmenkunde",
    IS_GROUP: "Gruppe",
    COMPANY_NAME: "Firma",
    GROUP_NAME: "Gruppenname",
    GROSS_AMOUNT: "Umsatz (€)",
    GROSS_PER_NIGHT: "€ / Nacht",
    CANCEL_PROBA: "Storno-Wahrscheinlichkeit",
    RISK_BUCKET: "Risiko",
}


def bucketize(proba: float) -> str:
    if proba >= HIGH_THR:
        return "high"
    if proba >= LOW_THR:
        return "uncertain"
    return "low"


def category_capacity(units_total: int) -> dict[str, int]:
    caps = {c: int(round(units_total * w)) for c, w in ROOM_CATEGORY_SPLIT.items()}
    diff = units_total - sum(caps.values())
    caps["BIG"] = max(0, caps["BIG"] + diff)
    return caps
