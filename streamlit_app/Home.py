"""Stayery Overbooking Analytics — Streamlit-Einstiegspunkt (Landing-Page).

Startseite: Hero, Datenstand, Navigations-Kacheln und Standort-Übersicht.

Start:  uv run streamlit run streamlit_app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "streamlit_app"))

import streamlit as st

import backend as B
from components import (
    benchmark_overbooking_allowance,
    hero,
    icon,
    inject_brand_css,
    load_locations,
    nav_card,
)

st.set_page_config(
    page_title="Stayery Overbooking Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_brand_css()

# Backend-Modus aus der Session (falls auf „Datenaktualisierung“ umgestellt).
if "backend_mode" in st.session_state:
    B.set_mode(st.session_state["backend_mode"])

hero(
    eyebrow="Stayery · Overbooking Analytics",
    title="Auslastung verstehen, Stornos vorhersagen, Overbooking steuern",
    subtitle="Wähl links oder unten einen Bereich. Die Heatmaps zeigen aktuelle "
    "Auslastung und prognostizierte Stornos je Standort — mit Tages-Breakdown "
    "der Anreisenden und Cancel-Wahrscheinlichkeit pro Buchung.",
)

# ============================== Datenstand ================================
st.subheader("Datenstand")

try:
    meta = B.get_metadata()
except Exception as e:  # noqa: BLE001
    meta = None
    st.warning(f"Datenstand konnte nicht geladen werden: {type(e).__name__}: {e}")

if meta:
    refreshed = str(meta.get("refreshed_at", "?"))[:19].replace("T", " ")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letzter Refresh", refreshed)
    c2.metric("Künftige Anreisen", f"{meta['upcoming']['rows']:,}".replace(",", "."))
    c3.metric("Ø Storno-Quote", f"{meta.get('cancel_rate', 0)*100:.0f} %",
              help="Durchschnittliche prognostizierte Storno-Wahrscheinlichkeit der künftigen Anreisen.")
    c4.metric("High-Risk-Buchungen", meta.get("high_risk", 0),
              help=f"Buchungen mit Storno-Wahrscheinlichkeit ≥ 75 %.")
    badge = "synthetisch (Platzhalter)" if meta.get("mode") == "dummy" else "echtes Modell"
    canceled = meta.get("canceled", {}).get("rows", 0)
    st.caption(f"Datenquelle: **{badge}** · {len(meta.get('properties', []))} Standorte · "
               f"{canceled:,} stornierte Buchungen ausgeschlossen · steuerbar unter „Datenaktualisierung“."
               .replace(",", "."))

st.divider()

# ============================== Bereiche (Nav-Cards) ======================
st.subheader("Bereiche")
st.caption("Vier Arbeitsbereiche — Sidebar-Filter, Heatmaps, Tabellen, CSV-Export.")

r1c1, r1c2 = st.columns(2, gap="medium")
with r1c1:
    nav_card(
        page="pages/1_Auslastung_und_Anreisen.py",
        icon=icon("calendar"),
        kicker="Dashboard",
        title="Auslastung & Anreisen",
        desc="Synthese der >90 %-Tage, Belegung / An- / Abreisen je Standort, "
        "Belegung je Zimmerkategorie für Upgrade-Entscheidungen und der "
        "Tages-Breakdown der Anreisenden.",
        link_label="→ Auslastung & Anreisen öffnen",
        status="ready",
    )
with r1c2:
    nav_card(
        page="pages/2_Overbooking_Predictions.py",
        icon=icon("trend"),
        kicker="Vorhersage",
        title="Overbooking Predictions",
        desc="Erwartete Stornos je Standort und Tag, Empfehlung gegen das "
        "Overbooking-Limit und die Cancel-Wahrscheinlichkeit pro Buchung — "
        "nach Tag filterbar.",
        link_label="→ Overbooking Predictions öffnen",
        status="ready",
    )

r2c1, r2c2 = st.columns(2, gap="medium")
with r2c1:
    nav_card(
        page="pages/4_Modell_und_Performance.py",
        icon=icon("bolt"),
        kicker="Modell",
        title="Modell & Performance",
        desc="Wie das Storno-Modell rechnet: Confusion-Matrix, ROC-AUC, "
        "Feature-Einfluss, Kalibrierung, historische Heatmap — plus Retrain.",
        link_label="→ Modell & Performance öffnen",
        status="ready",
    )
with r2c2:
    nav_card(
        page="pages/3_Datenaktualisierung.py",
        icon=icon("refresh"),
        kicker="Daten",
        title="Datenaktualisierung",
        desc="Snapshot-Status und Refresh per Knopfdruck. Umschalter zwischen "
        "Platzhalter-Backend und echtem Modell — ohne dass eine Seite "
        "angefasst werden muss.",
        link_label="→ Datenaktualisierung öffnen",
        status="ready",
    )

st.divider()

# ============================== Standorte =================================
st.subheader("Standorte")
st.caption(
    "Hotel-Stammdaten aus `configs/locations.yaml`. Overbooking-Limit nach "
    "Benchmark: unter 50 Units → 2 erlaubt, ab 50 Units → 4 erlaubt."
)

_loc = load_locations()
if _loc.empty:
    st.warning("Keine Standorte in `configs/locations.yaml` gefunden.")
else:
    import pandas as pd

    _table = pd.DataFrame({
        "Code": _loc["hotel_code"],
        "Stadt": _loc.get("city", ""),
        "Neighborhood": _loc.get("neighborhood").fillna("–") if "neighborhood" in _loc else "–",
        "Bundesland": _loc.get("bundesland", ""),
        "Units": _loc["units_total"],
        "Overbooking-Limit": _loc["units_total"].apply(benchmark_overbooking_allowance),
        "Eröffnet": _loc.get("opening_date").apply(
            lambda v: "TBD" if v in (None, "") or pd.isna(v) else str(v)
        ) if "opening_date" in _loc else "TBD",
    })
    st.dataframe(_table, hide_index=True, use_container_width=True, height=388)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Standorte gesamt", len(_loc))
    mc2.metric("Units gesamt", f"{int(_loc['units_total'].sum()):,}".replace(",", "."))
    mc3.metric("Overbookings erlaubt (Σ)",
               int(_loc["units_total"].apply(benchmark_overbooking_allowance).sum()))

st.divider()
st.caption("Stayery Overbooking Analytics · Design gespiegelt von RevenueBlindSpots.")
