# dash_app/backend/schema.py
# ---------------------------------------------------------------------------
# Canonical schema for scored bookings. PORTED VERBATIM (with comments) from
# streamlit_app/backend/schema.py so both the dummy and real backends emit
# EXACTLY these columns. Pages depend only on these constants — never on raw
# src/BigQuery column names — which is what lets us swap backends invisibly.
# We copy it (rather than import the Streamlit one) so dash_app stays a fully
# self-contained package per the project constraint.
# ---------------------------------------------------------------------------

from __future__ import annotations

# `Final` marks these as constants (type-checkers flag reassignment).
from typing import Final

# Risk-bucket thresholds. These are only UI fallbacks/initial slider values:
# the authoritative cut points are now derived per-model from the training
# artifacts via src.scoring.serving_thresholds() (low = validation base rate,
# high = F1-optimal threshold) and are baked into the scored parquet's
# `risk_bucket` column. Keep these in sync if you need a static default.
LOW_THR: Final[float] = 0.50     # below this => "low"
HIGH_THR: Final[float] = 0.50    # at/above this => "high"

# The three risk buckets and their English display labels.
RISK_BUCKETS: Final[tuple[str, ...]] = ("low", "uncertain", "high")
RISK_LABELS: Final[dict[str, str]] = {"low": "low", "uncertain": "uncertain", "high": "high"}
RISK_LABELS_DE = RISK_LABELS  # backwards-compatible alias (labels are now English)

# Synthetic-data vocabularies (used by dummy.py).
RATE_CATEGORIES: Final[tuple[str, ...]] = ("Flex", "Semi-Flex", "Non-Ref")
ROOM_CATEGORIES: Final[tuple[str, ...]] = ("BIG", "BIGGER", "UPPER", "UPPER AIR")
# Share of each room category within a property (sums to 1.0).
ROOM_CATEGORY_SPLIT: Final[dict[str, float]] = {
    "BIG": 0.42, "BIGGER": 0.33, "UPPER": 0.18, "UPPER AIR": 0.07,
}

# Booking status values.
STATUS_CONFIRMED: Final[str] = "Confirmed"
STATUS_CANCELED: Final[str] = "Canceled"

# ---- Column-name constants (the single source of truth for column keys) ----
# Pages reference S.HOTEL_CODE etc., never the literal string, so a rename is
# a one-line change here.
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

# The exact ordered set of columns every backend must return. Pages can rely on
# all of these existing.
COLUMNS: Final[tuple[str, ...]] = (
    BOOKING_ID, HOTEL_CODE, CITY, PROPERTY_NAME, STATUS,
    ARRIVAL, DEPARTURE, CREATED, ARRIVAL_DATE, DEPARTURE_DATE, DAYS_UNTIL_ARRIVAL,
    LEAD_TIME_DAYS, LOS_NIGHTS, STAY_BUCKET, ADULTS, UNIT_GROUP,
    RATE_PLAN, RATE_CATEGORY, IS_REFUNDABLE, IS_CANCELABLE, CANCELLATION_FEE,
    CHANNEL, TRAVEL_PURPOSE, GUARANTEE_TYPE, COUNTRY_CODE, IS_INTERNATIONAL,
    HAS_PROMO, HAS_CORPORATE_CODE, IS_CORPORATE, IS_GROUP, COMPANY_NAME, GROUP_NAME,
    GROSS_AMOUNT, GROSS_PER_NIGHT, CANCEL_PROBA, RISK_BUCKET,
)

# Display labels for table headers (used by the predictions table).
LABELS: Final[dict[str, str]] = {
    BOOKING_ID: "Booking",
    HOTEL_CODE: "Location",
    CITY: "City",
    PROPERTY_NAME: "Hotel",
    STATUS: "Status",
    ARRIVAL_DATE: "Arrival",
    DEPARTURE_DATE: "Departure",
    DAYS_UNTIL_ARRIVAL: "Days to arrival",
    LEAD_TIME_DAYS: "Lead time (days)",
    LOS_NIGHTS: "Nights",
    STAY_BUCKET: "Stay length",
    ADULTS: "Guests",
    UNIT_GROUP: "Room category",
    RATE_PLAN: "Rate plan",
    RATE_CATEGORY: "Rate category",
    IS_REFUNDABLE: "Refundable",
    IS_CANCELABLE: "Still cancelable",
    CANCELLATION_FEE: "Cancellation fee (€)",
    CHANNEL: "Channel",
    TRAVEL_PURPOSE: "Travel purpose",
    GUARANTEE_TYPE: "Guarantee",
    COUNTRY_CODE: "Country",
    IS_INTERNATIONAL: "International",
    HAS_PROMO: "Promo",
    HAS_CORPORATE_CODE: "Corporate code",
    IS_CORPORATE: "Corporate guest",
    IS_GROUP: "Group",
    COMPANY_NAME: "Company",
    GROUP_NAME: "Group name",
    GROSS_AMOUNT: "Revenue (€)",
    GROSS_PER_NIGHT: "€ / night",
    CANCEL_PROBA: "Cancellation probability",
    RISK_BUCKET: "Risk",
}
LABELS_DE = LABELS  # backwards-compatible alias (labels are now English)


def bucketize(proba: float) -> str:
    """Map one probability to a risk bucket — same rule as src.scoring.bucketize."""
    # >= HIGH_THR is high risk, >= LOW_THR (but below high) is uncertain, else low.
    if proba >= HIGH_THR:
        return "high"
    if proba >= LOW_THR:
        return "uncertain"
    return "low"


def category_capacity(units_total: int) -> dict[str, int]:
    """Split a property's total units across the four room categories.

    Multiplies the total by each category's share, then dumps the rounding
    remainder into BIG so the parts sum back to `units_total`.
    """
    # Round each category's share to an integer unit count.
    caps = {c: int(round(units_total * w)) for c, w in ROOM_CATEGORY_SPLIT.items()}
    # Rounding may lose/gain a unit or two; reconcile by adjusting BIG.
    diff = units_total - sum(caps.values())
    caps["BIG"] = max(0, caps["BIG"] + diff)
    return caps
