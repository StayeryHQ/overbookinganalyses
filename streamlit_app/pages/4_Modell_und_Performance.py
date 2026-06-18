from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "streamlit_app"))

import streamlit as st

import backend as B
from backend import model_perf as M
from components import alert_card, charts, hero, inject_brand_css, render_toc, section, ui

st.set_page_config(page_title="Modell & Performance · Stayery", layout="wide")
inject_brand_css()
charts.apply_style_once()

hero(
    eyebrow="Modell",
    title="Modell & Performance",
    subtitle="Wie das Storno-Modell rechnet, wie gut es trifft und worauf es achtet.",
)

card = M.model_card()

alert_card(
    "**Platzhalter-Modell mit echter Berechnung.** Eine kleine logistische Regression, "
    "auf dem synthetischen Datensatz trainiert und auf einem Holdout bewertet. Alle "
    "Metriken sind real gerechnet. Beim Wechsel auf das echte `src`-Modell bleibt diese "
    "Seite identisch — nur die Datenquelle wechselt.", kind="info")

render_toc([(1, "Steckbrief"), (2, "Performance"), (3, "Feature-Einfluss"),
            (4, "Kalibrierung"), (5, "Historie"), (6, "Retraining")])

# =============================================================================
with section(1, "Steckbrief"):
    c = st.columns(4)
    c[0].metric("Typ", "Logreg")
    c[1].metric("ROC-AUC", f"{card['auc']:.3f}", help="Trennschärfe; 0.5 = Zufall, 1.0 = perfekt.")
    c[2].metric("Basis-Storno-Rate", f"{card['base_rate']*100:.0f} %")
    c[3].metric("Features", card["n_features"])
    st.caption(f"Trainiert: {str(card['trained_at'])[:19].replace('T', ' ')} · "
               f"{card['n_train']:,} Train / {card['n_test']:,} Test · "
               f"Risiko-Schwellen: niedrig < {card['low_thr']:.0%} · hoch ≥ {card['high_thr']:.0%}"
               .replace(",", "."))

# =============================================================================
with section(2, "Performance", description="Trefferbild auf dem Holdout bei wählbarem Betriebspunkt."):
    thr = st.slider("Betriebs-Schwelle (Klassifikation Storno ja/nein)", 0.10, 0.60, 0.30, step=0.05,
                    help="Ab dieser vorhergesagten Wahrscheinlichkeit wird eine Buchung als "
                    "Storno klassifiziert. Bei niedriger Basisrate ist 0.5 oft zu streng.")
    perf = M.performance(thr)
    k = st.columns(5)
    k[0].metric("ROC-AUC", f"{perf['auc']:.3f}")
    k[1].metric("Accuracy", f"{perf['accuracy']*100:.0f} %")
    k[2].metric("Precision", f"{perf['precision']*100:.0f} %", help="Anteil echter Stornos unter den als Storno markierten.")
    k[3].metric("Recall", f"{perf['recall']*100:.0f} %", help="Anteil erkannter Stornos an allen tatsächlichen.")
    k[4].metric("F1", f"{perf['f1']:.2f}")
    cm1, cm2 = st.columns([1, 1])
    with cm1:
        charts.render(charts._sig("cm", thr), charts.confusion_fig, perf["confusion"])
    with cm2:
        ui.explain("Confusion-Matrix lesen",
                   "Zeilen = tatsächlich, Spalten = vorhergesagt. Oben-links und unten-rechts "
                   "sind korrekt. Unten-links (verpasste Stornos) ist beim Overbooking teurer "
                   "als oben-rechts — die Betriebs-Schwelle steuert diese Balance.")
        st.dataframe({"Metrik": ["True Negative", "False Positive", "False Negative", "True Positive"],
                      "Anzahl": [int(perf["confusion"][0, 0]), int(perf["confusion"][0, 1]),
                                 int(perf["confusion"][1, 0]), int(perf["confusion"][1, 1])]},
                     hide_index=True, use_container_width=True)

# =============================================================================
with section(3, "Feature-Einfluss", description="Was treibt die Storno-Wahrscheinlichkeit (SHAP-Proxy)."):
    ui.explain("Was ist das?",
               "Standardisierte Koeffizienten der Regression als Näherung für SHAP. "
               "Orange erhöht die Storno-Wahrscheinlichkeit, blau senkt sie. Im echten "
               "Modell stehen hier die SHAP-Werte aus dem Training.")
    charts.render(charts._sig("imp"), charts.importance_fig, M.feature_importance())

# =============================================================================
with section(4, "Kalibrierung", description="Stimmen vorhergesagte Wahrscheinlichkeiten mit der Realität überein."):
    ui.explain("Wofür wichtig?",
               "Fürs Overbooking zählt nicht nur die Rangfolge, sondern dass „30 %“ auch "
               "wirklich ~30 % bedeutet. Liegt die Kurve auf der Diagonalen, ist das Modell "
               "gut kalibriert — die erwarteten Stornos sind dann belastbar.")
    charts.render(charts._sig("calib"), charts.calibration_fig, M.calibration())

# =============================================================================
with section(5, "Historie", description="Tatsächliche Storno-Rate je Standort und Woche (Holdout)."):
    hm = M.history_matrix(B.hotel_labels())
    hm100 = hm * 100
    charts.render(charts._sig("hist"), charts.value_heatmap_fig, hm100,
                  cmap=charts._risk_cmap(), fmt="%.0f", vmax=float(hm100.values.max()) if hm100.size else 100,
                  x_labels=list(hm.columns))
    st.caption("Werte in % — dunkler = höhere realisierte Storno-Rate.")

# =============================================================================
with section(6, "Retraining", description="Modell auf dem aktuellen Buchungsbestand neu trainieren."):
    ui.explain("Was passiert hier?",
               "Im Platzhalter-Modus wird das Modell mit frischem Zufalls-Seed neu trainiert "
               "und alle Metriken oben aktualisieren sich. Im echten Modus stößt dieser Knopf "
               "das Training auf dem aktuellen Buchungsbestand an.")
    if st.button("Modell neu trainieren", type="primary"):
        with st.status("Trainiere …", expanded=False) as s:
            new = M.retrain()
            for k in list(st.session_state.keys()):
                if str(k).startswith("_ob_chart_cache"):
                    del st.session_state[k]
            s.update(label="Training abgeschlossen", state="complete")
        st.success(f"Neu trainiert · AUC {new['auc']:.3f} · "
                   f"{new['n_train']:,} Trainingszeilen.".replace(",", "."))
        st.rerun()

charts.style_collect()
st.page_link("Home.py", label="← Zurück zur Startseite")
