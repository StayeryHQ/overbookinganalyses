"""Datenaktualisierung — Snapshot-Status, Refresh, Backend-Umschalter."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "streamlit_app"))

import streamlit as st

import backend as B
from components import alert_card, hero, inject_brand_css

st.set_page_config(page_title="Datenaktualisierung · Stayery", layout="wide")
inject_brand_css()

# Backend-Modus aus der Session übernehmen (gilt prozessweit für alle Seiten).
if "backend_mode" in st.session_state:
    B.set_mode(st.session_state["backend_mode"])

hero(
    eyebrow="Daten",
    title="Datenaktualisierung",
    subtitle="Snapshot-Status ansehen und per Knopfdruck aktualisieren.",
)


def _clear_chart_cache() -> None:
    for k in list(st.session_state.keys()):
        if str(k).startswith("_ob_chart_cache"):
            del st.session_state[k]


# =============================================================================
st.subheader("Aktueller Snapshot")
meta = B.get_metadata()
mode = meta.get("mode", "dummy")

refreshed = str(meta.get("refreshed_at", "?"))[:19].replace("T", " ")
win = meta.get("window", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Letzter Refresh", refreshed)
c2.metric("Bestätigt", f"{meta.get('confirmed', {}).get('rows', 0):,}".replace(",", "."))
c3.metric("Storniert (raus)", f"{meta.get('canceled', {}).get('rows', 0):,}".replace(",", "."))
c4.metric("Anreisen (künftig)", f"{meta['upcoming']['rows']:,}".replace(",", "."))
st.caption(
    f"Anreise-Range: **{win.get('earliest','?')}** bis **{win.get('latest','?')}** · "
    f"{len(meta.get('properties', []))} Standorte · Modus: **{mode}**"
)

st.divider()

# =============================================================================
st.subheader("Aktualisieren")

if mode == "dummy":
    alert_card(
        "**Dummy-Modus.** Der Refresh erzeugt einen neuen synthetischen Snapshot "
        "(anderer Seed) — ideal zum Entwickeln und Zeigen der App, ohne echtes "
        "Modell. Auf „real“ umstellen, sobald Modell & Daten stehen.",
        kind="info",
    )
else:
    alert_card(
        "**Real-Modus.** Der Refresh zieht zukünftige Buchungen aus BigQuery und "
        "bewertet sie mit dem Modell (`src.score_upcoming`). Voraussetzung: "
        "trainiertes Modell auf Disk und Google-Cloud-Auth "
        "(`gcloud auth application-default login`).",
        kind="warning",
    )

if st.button("Daten aktualisieren", type="primary",
             help="Dummy: neuer Snapshot. Real: BigQuery-Pull (nur Zukunft) + Scoring."):
    with st.status("Aktualisiere …", expanded=True) as status:
        try:
            st.write("Starte Refresh im Modus: " + mode)
            new_meta = B.refresh()
            _clear_chart_cache()
            status.update(label="Refresh abgeschlossen", state="complete")
            st.success(
                f"Fertig. {new_meta['reservations']['rows']:,} Buchungen, "
                f"{new_meta['upcoming']['rows']:,} künftige Anreisen, "
                f"{len(new_meta['properties'])} Standorte."
                .replace(",", ".")
            )
        except Exception as e:  # noqa: BLE001
            status.update(label="Refresh fehlgeschlagen", state="error")
            st.error(f"**{type(e).__name__}**: {e}")
            with st.expander("Details"):
                import traceback
                st.code(traceback.format_exc())

st.divider()

# =============================================================================
st.subheader("Backend-Modus")
st.caption(
    "Die Seiten kennen nur ein einheitliches Schema — der Wechsel zwischen "
    "synthetischem Platzhalter und echtem Modell ist ein reiner Backend-Tausch."
)

choice = st.radio(
    "Quelle der Bewertungen",
    options=["dummy", "real"],
    index=0 if mode == "dummy" else 1,
    format_func=lambda m: ("Dummy-Modell (synthetisch, Standard)"
                           if m == "dummy" else "Echtes Modell (src.score_upcoming)"),
    horizontal=True,
)
if choice != mode:
    if st.button(f"Auf „{choice}“ umstellen"):
        st.session_state["backend_mode"] = choice
        B.set_mode(choice)
        _clear_chart_cache()
        st.rerun()

st.info(
    "Alternativ dauerhaft per Umgebungsvariable: `OVERBOOKING_BACKEND=real` vor "
    "dem Start setzen.",
    icon=None,
)

st.page_link("Home.py", label="← Zurück zur Startseite")
