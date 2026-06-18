from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "streamlit_app"))

import pandas as pd
import streamlit as st

import backend as B
from backend import derive as D
from backend import schema as S
from components import alert_card, charts, hero, inject_brand_css, render_toc, section, tables, ui

st.set_page_config(page_title="Overbooking Predictions · Stayery", layout="wide")
inject_brand_css()
charts.apply_style_once()

if "backend_mode" in st.session_state:
    B.set_mode(st.session_state["backend_mode"])

hero(
    eyebrow="Vorhersage",
    title="Overbooking Predictions",
    subtitle="Erwartete Stornos je Standort und Tag — als Grundlage für sichere "
    "Overbooking-Entscheidungen.",
)

bookings = B.get_scored_bookings()
units = B.units_by_hotel()
labels = B.hotel_labels()
today = pd.Timestamp.today().normalize()

if bookings.empty:
    alert_card("Keine Daten. Bitte auf „Datenaktualisierung“ einen Snapshot erzeugen.", kind="warning")
    st.stop()

f = ui.sidebar_filters(bookings, units, labels, prefix="pred",
                       want_window=True, want_threshold=True, want_rooms=True)
base = ui.base_frame(bookings, f)
filtered = ui.apply_filters(bookings, f)
n_canceled = ui.canceled_count(bookings, f)

codes = f["codes"]
threshold = f["threshold"]
dates = pd.date_range(today, periods=f["horizon"], freq="D")
sel_units = {c: units[c] for c in codes}

st.sidebar.caption(f"{n_canceled:,} stornierte ausgeschlossen".replace(",", "."))

ui.explain("Wie sind die Zahlen zu lesen?",
           f"Das Modell schätzt je Buchung eine **Storno-Wahrscheinlichkeit**.\n\n"
           f"- **Erwartete Stornos** = Summe dieser Wahrscheinlichkeiten. Das ist die "
           f"statistisch korrekte Grundlage fürs Overbooking — auch wenn keine einzelne "
           f"Buchung „sicher“ storniert, summieren sich viele kleine Wahrscheinlichkeiten.\n"
           f"- **High-Risk** = Einzelbuchungen mit Wahrscheinlichkeit ≥ {threshold:.0%} "
           f"(Schwelle links einstellbar) — für die manuelle Prüfung.\n\n"
           f"Risiko-Stufen: niedrig < {S.LOW_THR:.0%} · unsicher {S.LOW_THR:.0%}–{S.HIGH_THR:.0%} "
           f"· hoch ≥ {S.HIGH_THR:.0%}. Bereits stornierte Buchungen sind ausgeschlossen.")

render_toc([(1, "Prognose-Heatmap"), (2, "Empfehlung je Standort"), (3, "Buchungen")])

# =============================================================================
with section(1, "Prognose-Heatmap", description="Erwartete Stornos (Σ Wahrscheinlichkeit) je Standort und Tag."):
    pm = D.prediction_matrix(base, dates, sel_units, threshold, value="expected_cancels")
    pm.index = [labels.get(c, c) for c in pm.index]
    charts.render(charts.heatmap_signature(pm, "pred", threshold), charts.cancellation_heatmap_fig,
                  pm, integer=False)

    pl = D.prediction_long(base, dates, sel_units, threshold)
    win = base[base[S.ARRIVAL_DATE].isin(dates)]
    m = st.columns(4)
    m[0].metric("Anreisen im Fenster", f"{int(pl['arrivals'].sum()):,}".replace(",", "."))
    m[1].metric("Erwartete Stornos", f"{pl['expected_cancels'].sum():.0f}",
                help="Summe der Storno-Wahrscheinlichkeiten über alle gewählten Standorte "
                "und Tage des Fensters.")
    m[2].metric(f"High-Risk (≥ {threshold:.0%})", int(pl["high_risk"].sum()))
    m[3].metric("Ø Storno-Quote", f"{win[S.CANCEL_PROBA].mean()*100:.0f} %" if len(win) else "–",
                help="Durchschnittliche prognostizierte Storno-Wahrscheinlichkeit der Anreisen im Fenster.")

# =============================================================================
with section(2, "Empfehlung je Standort",
             description="Erwartete Stornos gegen das Overbooking-Limit (< 50 Units → 2, ≥ 50 → 4)."):
    ui.explain("Wie entsteht die Empfehlung?",
               "Vorläufiger Mechanismus: empfohlenes Overbooking = min(Limit, "
               "gerundete erwartete Stornos pro Tag). So bleibt man im Rahmen des Limits, "
               "nutzt aber den statistischen Storno-Puffer. Den Mechanismus bauen wir später aus.")
    rec = D.recommendation_table(base, dates, sel_units, threshold, labels)
    st.dataframe(
        rec, hide_index=True, use_container_width=True,
        column_config={
            "Empfehlung": st.column_config.NumberColumn("Empfehlung", help="Empfohlenes Overbooking pro Tag", format="%d"),
            "Erwartete Stornos": st.column_config.NumberColumn(format="%.1f"),
            "Ø / Tag": st.column_config.NumberColumn(format="%.1f"),
        })

# =============================================================================
with section(3, "Buchungen", description="Cancel-Wahrscheinlichkeit pro Buchung — nach Tag filterbar."):
    g1, g2, g3 = st.columns([1.6, 1.2, 1.2])
    with g1:
        wd = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        day_opt = st.selectbox("Anreisetag", ["Alle Tage", *list(dates)],
                               format_func=lambda d: d if isinstance(d, str)
                               else f"{wd[d.weekday()]} {d.day:02d}.{d.month:02d}.", key="pred_day")
    with g2:
        only_high = st.toggle(f"Nur High-Risk (≥ {threshold:.0%})", value=False)
    with g3:
        sort_desc = st.toggle("Höchstes Risiko oben", value=True)

    day = None if day_opt == "Alle Tage" else day_opt
    tbl = D.predictions_table(filtered, dates=dates, day=day)
    if only_high:
        tbl = tbl[tbl[S.CANCEL_PROBA] >= threshold]
    tbl = tbl.sort_values(S.CANCEL_PROBA, ascending=not sort_desc)

    exp = tbl[S.CANCEL_PROBA].sum()
    st.caption(f"{len(tbl):,} Buchungen · erwartete Stornos in Auswahl: {exp:.1f}".replace(",", "."))
    disp, cfg = tables.booking_view(tbl)
    st.dataframe(disp, hide_index=True, use_container_width=True, height=460, column_config=cfg)
    st.download_button("Als CSV herunterladen", data=tbl.to_csv(index=False).encode("utf-8"),
                       file_name="overbooking_predictions.csv", mime="text/csv", key="dl_pred")

charts.style_collect()
st.page_link("Home.py", label="← Zurück zur Startseite")
