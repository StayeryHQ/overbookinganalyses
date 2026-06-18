from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "streamlit_app"))

import numpy as np
import pandas as pd
import streamlit as st

import backend as B
from backend import derive as D
from backend import schema as S
from components import alert_card, charts, hero, inject_brand_css, render_toc, section, tables, ui

st.set_page_config(page_title="Auslastung & Anreisen · Stayery", layout="wide")
inject_brand_css()
charts.apply_style_once()

if "backend_mode" in st.session_state:
    B.set_mode(st.session_state["backend_mode"])

hero(
    eyebrow="Dashboard",
    title="Auslastung & Anreisen",
    subtitle="Wo wird es eng, wer reist an, wo lässt sich upgraden — Belegung, "
    "An- und Abreisen je Standort und Zimmerkategorie.",
)

bookings = B.get_scored_bookings()
units = B.units_by_hotel()
labels = B.hotel_labels()
caps = B.capacity_by_category()
today = pd.Timestamp.today().normalize()

if bookings.empty:
    alert_card("Keine Daten. Bitte auf „Datenaktualisierung“ einen Snapshot erzeugen.", kind="warning")
    st.stop()

f = ui.sidebar_filters(bookings, units, labels, prefix="aus", want_window=True, want_rooms=True)
base = ui.base_frame(bookings, f)
filtered = ui.apply_filters(bookings, f)
n_canceled = ui.canceled_count(bookings, f)

codes = f["codes"]
dates = pd.date_range(today, periods=f["horizon"], freq="D")
sel_units = {c: units[c] for c in codes}

st.sidebar.caption(
    f"{len(base):,} bestätigte Buchungen · {n_canceled:,} stornierte ausgeschlossen"
    .replace(",", ".")
)

render_toc([(1, "Synthese"), (2, "Belegung & Bewegungen"), (3, "Zimmerkategorie"), (4, "Anreisen-Detail")])

# =============================================================================
with section(1, "Synthese", description="Der Morgen-Blick: wo wird es eng und was bedeutet das."):
    sc1, sc2 = st.columns([3, 1])
    with sc2:
        occ_thr = st.slider("Auslastungs-Schwelle", 80, 100, 90, step=1,
                            help="Tage, an denen ein Standort über dieser Belegung liegt.") / 100
    occ = D.occupancy_matrix(base, dates, sel_units)
    vals = occ.values
    syn = D.synthesis_by_day(base, dates, sel_units, occ_thr, S.HIGH_THR)

    with sc1:
        m = st.columns(4)
        m[0].metric("Ø Belegung", f"{np.nanmean(vals)*100:.0f} %" if vals.size else "–")
        m[1].metric(f"Tage > {occ_thr:.0%}", int(syn["Standorte"].gt(0).sum()) if len(syn) else 0)
        m[2].metric("Overbooking-Nächte", int((vals > 1.0).sum()) if vals.size else 0,
                    help="Belegung über 100 % (mehr bestätigte Buchungen als Units).")
        m[3].metric("Stornierte raus", f"{n_canceled:,}".replace(",", "."))

    ui.explain("Wie lese ich das?",
               "Belegung = bestätigte Buchungen, die in einer Nacht ein Zimmer belegen, "
               "geteilt durch die Kapazität. **Bereits stornierte Buchungen sind "
               "ausgeschlossen.** Die Tabelle zeigt nur Tage über der Schwelle — dort lohnt "
               "der Blick auf Upgrades, Overbooking und mögliche Stornos.")

    if len(syn):
        syn_disp = syn.copy()
        syn_disp["Datum"] = syn_disp["Datum"].dt.strftime("%a %d.%m.")
        syn_disp["Wo"] = syn_disp["Wo"].apply(
            lambda s: ", ".join(labels.get(c, c) for c in s.split(", ")))
        syn_disp["Ø Belegung"] = (syn_disp["Ø Belegung"] * 100).round(0).astype(int).astype(str) + " %"
        st.dataframe(syn_disp, hide_index=True, use_container_width=True)
    else:
        alert_card(f"Kein Standort über {occ_thr:.0%} im gewählten Zeitfenster.", kind="success")

# =============================================================================
with section(2, "Belegung & Bewegungen", description="Belegung, Anreisen und Abreisen je Standort."):
    ui.explain("Wie lese ich das?",
               "Drei Sichten auf dieselben Tage: **Belegung** in % der Kapazität "
               "(rot umrandet = Overbooking), **Anreisen** und **Abreisen** als Anzahl "
               "Buchungen pro Tag. Wochenenden sind in der Datumszeile hervorgehoben.")
    t1, t2, t3 = st.tabs(["Belegung", "Anreisen", "Abreisen"])
    occ_lab = occ.copy()
    occ_lab.index = [labels.get(c, c) for c in occ_lab.index]
    with t1:
        charts.render(charts.heatmap_signature(occ_lab, "occ"), charts.occupancy_heatmap_fig, occ_lab)
    with t2:
        am = D.arrivals_matrix(base, dates, codes)
        am.index = [labels.get(c, c) for c in am.index]
        charts.render(charts.heatmap_signature(am, "arr"), charts.value_heatmap_fig, am,
                      cmap=charts.arrivals_cmap())
    with t3:
        dm = D.departures_matrix(base, dates, codes)
        dm.index = [labels.get(c, c) for c in dm.index]
        charts.render(charts.heatmap_signature(dm, "dep"), charts.value_heatmap_fig, dm,
                      cmap=charts.departures_cmap())

# =============================================================================
with section(3, "Zimmerkategorie", description="Belegung je Kategorie — die Basis für Upgrade-Entscheidungen."):
    ui.explain("Wofür?",
               "Ist eine Kategorie (z. B. BIG) ausgebucht, während eine höhere (UPPER / "
               "UPPER AIR) noch frei ist, kann man gezielt upgraden. Rot umrandet = "
               "ausgebucht (≥ 100 %).")
    cat_code = st.selectbox("Standort", codes, format_func=lambda c: labels.get(c, c), key="cat_loc")
    com = D.category_occupancy_matrix(base, dates, cat_code, caps[cat_code])
    charts.render(charts.heatmap_signature(com, "cat", cat_code), charts.occupancy_heatmap_fig, com)

    sold = [(cat, d) for cat in com.index for d in com.columns if com.loc[cat, d] >= 1.0]
    free_high = {cat: float(com.loc[cat].mean()) for cat in com.index}
    cap_line = " · ".join(f"{cat}: {caps[cat_code][cat]}" for cat in S.ROOM_CATEGORIES)
    st.caption(f"Kapazität {labels.get(cat_code, cat_code)} — {cap_line}")
    if sold:
        days = sorted({d for _, d in sold})
        worst = ", ".join(f"{cat} ({d.strftime('%d.%m.')})" for cat, d in sold[:8])
        alert_card(f"**{len(sold)} ausgebuchte Kategorie-Tage.** z. B. {worst}"
                   + (" …" if len(sold) > 8 else "")
                   + " — Upgrade in eine freiere, höhere Kategorie prüfen.", kind="warning")
    else:
        alert_card("Keine Kategorie im Fenster ausgebucht.", kind="success")

# =============================================================================
with section(4, "Anreisen-Detail", description="Standort und Tag wählen — wer reist an."):
    dc1, dc2 = st.columns(2)
    with dc1:
        d_code = st.selectbox("Standort", codes, format_func=lambda c: labels.get(c, c), key="det_loc")
    loc_arr = base[base[S.HOTEL_CODE] == d_code]
    counts = {d: int((loc_arr[S.ARRIVAL_DATE] == d).sum()) for d in dates}
    wd = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    with dc2:
        d_day = st.selectbox("Anreisetag", list(dates),
                             format_func=lambda d: f"{wd[d.weekday()]} {d.day:02d}.{d.month:02d}. — {counts[d]} Anreisen",
                             key="det_day")

    sub = D.arrivals(filtered, d_code, d_day)
    k = D.arrivals_kpis(sub)
    if k["n"] == 0:
        alert_card(f"Keine (gefilterten) Anreisen am {d_day.date()} in {labels.get(d_code, d_code)}.", kind="info")
    else:
        r1 = st.columns(4)
        r1[0].metric("Anreisen", k["n"])
        r1[1].metric("Noch stornierbar", k["cancelable"], help="Erstattbarer Ratenplan, Frist offen.")
        r1[2].metric("Non-Refundable", k["non_refundable"])
        r1[3].metric("Long-Stay (≥ 8 N.)", k["long_stay"])
        r2 = st.columns(4)
        r2[0].metric("Ø Lead-Time", f"{k['avg_lead_time']:.0f} T")
        r2[1].metric("Firmenkunden", k["corporate"])
        r2[2].metric("Gruppen", k["group"])
        r2[3].metric("Ø Storno-Wkt.", f"{k['avg_cancel_proba']*100:.0f} %")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Lead-Time-Verteilung")
            charts.render(charts._sig(d_code, d_day, "lt"), charts.hist_fig, sub[S.LEAD_TIME_DAYS],
                          xlabel="Lead-Time (Tage)")
        with cc2:
            st.caption("Storno-Wahrscheinlichkeit")
            charts.render(charts._sig(d_code, d_day, "pr"), charts.hist_fig, sub[S.CANCEL_PROBA] * 100,
                          xlabel="Storno-Wahrscheinlichkeit (%)", color="#EB6E14")

        st.markdown("#### Breakdown")
        b1, b2, b3 = st.columns(3)
        b1.caption("Nach Raten-Kategorie")
        b1.dataframe(D.breakdown_by(sub, S.RATE_CATEGORY), hide_index=True, use_container_width=True)
        b2.caption("Nach Zimmerkategorie")
        b2.dataframe(D.breakdown_by(sub, S.UNIT_GROUP), hide_index=True, use_container_width=True)
        b3.caption("Nach Kanal")
        b3.dataframe(D.breakdown_by(sub, S.CHANNEL), hide_index=True, use_container_width=True)

        disp, cfg = tables.booking_view(sub)
        charts.data_table_expander(
            disp, title=f"Alle Anreisen am {d_day.day:02d}.{d_day.month:02d}. ({labels.get(d_code, d_code)})",
            filename=f"anreisen_{d_code}_{d_day.date()}", column_config=cfg, expanded=True, height=420)

charts.style_collect()
st.page_link("Home.py", label="← Zurück zur Startseite")
