# dash_app/pages/modell_performance.py
# ---------------------------------------------------------------------------
# "Modell & Performance" — the full model page. Mirrors
# streamlit_app/pages/4_Modell_und_Performance.py:
#   * a model selector (dropdown) + an operating-threshold slider;
#   * a model-card metric row (AUC / base rate / #features / AP);
#   * a confusion matrix (plotting.confusion_fig);
#   * ROC + PR + calibration curves (plotting.roc_curve_fig / pr_curve_fig /
#     calibration_fig);
#   * a feature-importance bar chart (plotting.horizontal_importance).
#
# Data comes from dash_app.backend.model_perf, which reads the real model card
# (reports/tables/<model>/model_card.json) in real mode and trains a tiny numpy
# logreg on the synthetic snapshot in dummy mode — same SHAPE either way.
# ---------------------------------------------------------------------------

from __future__ import annotations

import dash
from dash import callback, dcc, html, Input, Output

# Backend facade (for the current mode) + the model-perf data layer + UI + config.
from dash_app import backend as B
from dash_app.backend import model_perf as MP
from dash_app.components import alert, explain, hero, metric_row, section, caption
from dash_app import config as CFG

# Plotly chart factories from the shared module.
from src.plotting import (
    calibration_fig,
    confusion_fig,
    horizontal_importance,
    pr_curve_fig,
    roc_curve_fig,
)

# Register at /modell-performance.
dash.register_page(
    __name__,
    path="/modell-performance",
    name="Modell & Performance",
    title="Modell & Performance · Stayery",
    order=CFG.PAGE_ORDER["modell_performance"],
)

# Element ids.
_MODEL_DD = "perf-model-dropdown"
_THR_SLIDER = "perf-threshold-slider"
_CARD_ROW = "perf-card-row"
_PERF_ROW = "perf-metric-row"
_CONFUSION = "perf-confusion-graph"
_ROC = "perf-roc-graph"
_PR = "perf-pr-graph"
_CALIB = "perf-calib-graph"
_IMPORTANCE = "perf-importance-graph"
_NOTE = "perf-curve-note"


def _model_options() -> list[dict]:
    """Dropdown options: 'auto' + each available/known model (same as predictions)."""
    opts = [{"label": "Automatisch (bester AUC)", "value": ""}]
    avail = B.available_models()
    names = avail if avail else list(CFG.MODEL_LABELS.keys())
    for name in names:
        opts.append({"label": CFG.MODEL_LABELS.get(name, name), "value": name})
    return opts


def _build_views(model_value: str, threshold: float):
    """Compute all dynamic outputs for a model + threshold selection.

    Returns: (card_row, perf_row, confusion_fig, roc_fig, pr_fig, calib_fig,
    importance_fig, note). Used by the initial layout AND the callback.
    """
    # Resolve the current mode + selected model name (""/None => auto).
    mode = B.mode()
    model_name = model_value or None

    # ---- Model card metric row -------------------------------------------
    card = MP.model_card(mode, model_name)
    card_row = metric_row([
        {"label": "Modell", "value": card["name"]},
        {"label": "ROC-AUC", "value": f"{card['auc']:.3f}",
         "help": "Trennschärfe; 0.5 = Zufall, 1.0 = perfekt."},
        {"label": "Basis-Storno-Rate", "value": f"{card['base_rate'] * 100:.0f} %"},
        {"label": "Features", "value": card["n_features"]},
    ])

    # ---- Performance-at-threshold metric row + confusion matrix ----------
    perf = MP.performance(mode, model_name, threshold)
    perf_row = metric_row([
        {"label": "Accuracy", "value": f"{perf['accuracy'] * 100:.0f} %"},
        {"label": "Precision", "value": f"{perf['precision'] * 100:.0f} %",
         "help": "Anteil echter Stornos unter den als Storno markierten."},
        {"label": "Recall", "value": f"{perf['recall'] * 100:.0f} %",
         "help": "Anteil erkannter Stornos an allen tatsächlichen."},
        {"label": "F1", "value": f"{perf['f1']:.2f}"},
    ])
    # plotting.confusion_fig(cm) -> annotated 2x2 heatmap on the brand risk ramp.
    cm_fig = confusion_fig(perf["confusion"])

    # ---- ROC / PR / calibration curves -----------------------------------
    roc = MP.roc_curve(mode, model_name)
    pr = MP.pr_curve(mode, model_name)
    cal = MP.calibration(mode, model_name)
    # Each factory takes a {label: (...)} dict and returns a brand-themed figure.
    roc_fig = roc_curve_fig(roc["curves"], title="ROC")
    pr_fig = pr_curve_fig(pr["curves"], base_rate=pr.get("base_rate"), title="Precision-Recall")
    calib_fig_obj = calibration_fig(cal["curves"], title="Kalibrierung")

    # ---- Feature importance ----------------------------------------------
    labels_, values_ = MP.feature_importance(mode, model_name)
    # plotting.horizontal_importance: orange = increases risk, blue = decreases.
    # (Real gains are all positive => all orange, which is fine.)
    imp_fig = horizontal_importance(labels_, values_, title="Feature-Einfluss",
                                    xaxis_title="Beitrag / Gain")

    # ---- Caveat note when curve points aren't persisted on disk (real mode) --
    if mode == "real" and not (roc.get("curves_available", True)
                               and pr.get("curves_available", True)
                               and cal.get("curves_available", True)):
        note = alert(
            "Hinweis: Im Real-Modus sind nur die Kennzahlen (AUC, AP, Brier) und die "
            "Operating-Points im model_card.json gespeichert — die vollständigen "
            "ROC-/PR-/Kalibrierungs-Punktwolken liegen nicht auf Disk. Die Kurven hier "
            "sind daher Näherungen aus den Operating-Points. Persistiere die "
            "Test-Predictions, um echte Kurven zu zeigen.",
            kind="warning",
        )
    else:
        note = html.Div()  # empty placeholder in dummy mode

    return card_row, perf_row, cm_fig, roc_fig, pr_fig, calib_fig_obj, imp_fig, note


# Map model registry name -> the reports/figures/<dir> the model notebook writes to.
_FIG_DIRS = {"logreg": "01_logreg", "xgboost": "02_xgboost", "histgb": "03_histgb"}
# XAI plots the notebooks keep as STATIC PNG (no clean Plotly equivalent), embedded
# via the /figures Flask route registered in app.py.
_XAI_PNGS = [
    ("11_shap_beeswarm.png",  "SHAP Beeswarm — Beitrag jedes Features über viele Buchungen"),
    ("11_shap_waterfall.png", "SHAP Waterfall — eine einzelne Buchung komplett zerlegt"),
    ("11_pdp_ice.png",        "PDP + ICE — Effektkurven der wichtigsten Features"),
]


def _xai_gallery() -> list:
    """Build the XAI image gallery from reports/figures/<model>/*.png (if present).

    SHAP/PDP have no clean Plotly form, so the model notebooks save them as PNGs;
    here we embed whatever exists as <img> (served by app.py's /figures route).
    If nothing is on disk yet (notebooks not run), show a hint instead.
    """
    figures_root = CFG.REPO_ROOT / "reports" / "figures"     # <repo>/reports/figures
    blocks = []                                              # one block per model with figures
    for name, figdir in _FIG_DIRS.items():                   # each known model
        # Keep only the XAI PNGs that actually exist for this model.
        present = [(f, cap) for f, cap in _XAI_PNGS if (figures_root / figdir / f).exists()]
        if not present:
            continue                                         # nothing for this model yet
        imgs = []
        for fname, cap in present:                           # each available XAI image
            imgs.append(html.Figure([
                # <img src="/figures/<model>/<file>.png"> — streamed by the Flask route.
                html.Img(src=f"/figures/{figdir}/{fname}",
                         style={"width": "100%", "maxWidth": "720px",
                                "border": "1px solid #ECEAE0", "borderRadius": "8px"}),
                html.Figcaption(cap, className="stayery-caption"),  # image caption
            ], style={"margin": "0 0 1.2rem 0"}))
        # Group the model's images under its friendly label.
        blocks.append(html.Div([html.H4(CFG.MODEL_LABELS.get(name, name),
                                         className="stayery-h4"), *imgs]))
    if not blocks:                                           # no figures on disk at all
        return [html.Div("Die SHAP-/PDP-Grafiken erscheinen hier, sobald 01/02 gelaufen sind "
                         "(sie werden als statische Bilder aus reports/figures/ eingebettet).",
                         className="stayery-caption")]
    return blocks


def layout(**kwargs) -> html.Div:
    """Page layout factory — builds the initial view (auto model, threshold 0.30)."""
    init_thr = 0.30
    (card_row, perf_row, cm_fig, roc_fig, pr_fig,
     calib_fig_obj, imp_fig, note) = _build_views("", init_thr)

    return html.Div([
        hero(
            eyebrow="Modell",
            title="Modell & Performance",
            subtitle="Wie das Storno-Modell rechnet, wie gut es trifft und worauf es achtet.",
        ),

        # Model selector (same control as the predictions page).
        html.Div([
            html.Div("Modell", className="stayery-control-label"),
            dcc.Dropdown(id=_MODEL_DD, options=_model_options(), value="",
                         clearable=False, style={"minWidth": "280px"}),
        ], className="stayery-controls"),

        # Live caveat note (only populated in real mode without persisted curves).
        html.Div(note, id=_NOTE),

        # Section 1: model steckbrief.
        section(1, "Steckbrief", children=[html.Div(card_row, id=_CARD_ROW)]),

        # Section 2: performance at a chosen operating threshold + confusion.
        section(2, "Performance",
                description="Trefferbild bei wählbarem Betriebspunkt.",
                children=[
                    html.Div([
                        html.Div("Betriebs-Schwelle (Storno ja/nein)", className="stayery-control-label"),
                        # dcc.Slider: a draggable threshold control. marks label key points.
                        dcc.Slider(id=_THR_SLIDER, min=0.10, max=0.60, step=0.05, value=init_thr,
                                   marks={i / 100: f"{i/100:.2f}" for i in range(10, 61, 10)}),
                    ], style={"maxWidth": "520px", "marginBottom": "0.8rem"}),
                    html.Div(perf_row, id=_PERF_ROW),
                    explain("Confusion-Matrix lesen",
                            "Zeilen = tatsächlich, Spalten = vorhergesagt. Unten-links "
                            "(verpasste Stornos) ist beim Overbooking teurer als oben-rechts "
                            "— die Betriebs-Schwelle steuert diese Balance."),
                    dcc.Graph(id=_CONFUSION, figure=cm_fig, config={"displayModeBar": False}),
                ]),

        # Section 3: ROC + PR side by side.
        section(3, "Trennschärfe (ROC & PR)",
                description="ROC und Precision-Recall auf dem Holdout / Testset.",
                children=[
                    html.Div([
                        # Two graphs in a flex row so they sit side by side on wide screens.
                        html.Div(dcc.Graph(id=_ROC, figure=roc_fig, config={"displayModeBar": False}),
                                 style={"flex": "1 1 380px", "minWidth": "0"}),
                        html.Div(dcc.Graph(id=_PR, figure=pr_fig, config={"displayModeBar": False}),
                                 style={"flex": "1 1 380px", "minWidth": "0"}),
                    ], style={"display": "flex", "flexWrap": "wrap", "gap": "1rem"}),
                ]),

        # Section 4: calibration.
        section(4, "Kalibrierung",
                description="Stimmen vorhergesagte Wahrscheinlichkeiten mit der Realität überein.",
                children=[
                    explain("Wofür wichtig?",
                            "Fürs Overbooking zählt, dass „30 %“ auch ~30 % bedeutet. Liegt "
                            "die Kurve auf der Diagonalen, sind die erwarteten Stornos belastbar."),
                    dcc.Graph(id=_CALIB, figure=calib_fig_obj, config={"displayModeBar": False}),
                ]),

        # Section 5: feature importance.
        section(5, "Feature-Einfluss",
                description="Was treibt die Storno-Wahrscheinlichkeit.",
                children=[
                    dcc.Graph(id=_IMPORTANCE, figure=imp_fig, config={"displayModeBar": False}),
                ]),

        # Section 6: "under the hood" — what the model + data are, plus the SHAP/PDP
        # images (static PNGs from the notebooks; no clean Plotly equivalent).
        section(6, "Unter der Haube — Erklärbarkeit (XAI)",
                description="Wie das Modell zu seiner Einschätzung kommt — und worauf die Daten beruhen.",
                children=[
                    explain("Welche Daten?",
                            "Trainiert auf historischen, aufgelösten Buchungen (Notebook 00 → Roster + "
                            "Point-in-Time-Split). Features sind ausschließlich zur Buchungszeit bekannte "
                            "Größen: Lead-Time, Preis, Kanal, Rate, Zimmer, Saison usw. Leakage-Felder "
                            "(Firma/Adresse/Profil) sind bewusst ausgeschlossen — siehe 00 §7.5."),
                    explain("SHAP & PDP lesen",
                            "SHAP zerlegt jede Vorhersage additiv in Feature-Beiträge (Beeswarm = global "
                            "über viele Buchungen, Waterfall = eine einzelne Buchung). PDP/ICE zeigen, wie "
                            "sich das Risiko ändert, wenn man ein Feature variiert. Beide stammen direkt aus "
                            "den Modell-Notebooks (statische Bilder)."),
                    # The gallery embeds whatever XAI PNGs exist on disk.
                    html.Div(_xai_gallery()),
                ]),
    ])


# ---------------------------------------------------------------------------
# Callback: recompute everything when the model OR the threshold changes.
#   * Inputs : dropdown value + slider value.
#   * Outputs: both metric rows, all four figures, and the caveat note.
# Threshold only affects the confusion/performance pieces, but recomputing the
# lot keeps the wiring simple and the page consistent.
# ---------------------------------------------------------------------------
@callback(
    Output(_CARD_ROW, "children"),
    Output(_PERF_ROW, "children"),
    Output(_CONFUSION, "figure"),
    Output(_ROC, "figure"),
    Output(_PR, "figure"),
    Output(_CALIB, "figure"),
    Output(_IMPORTANCE, "figure"),
    Output(_NOTE, "children"),
    Input(_MODEL_DD, "value"),
    Input(_THR_SLIDER, "value"),
)
def _on_change(model_value: str, threshold: float):
    """Rebuild all dynamic outputs for the selected model + threshold."""
    (card_row, perf_row, cm_fig, roc_fig, pr_fig,
     calib_fig_obj, imp_fig, note) = _build_views(model_value, threshold)
    # Return in the SAME order as the Output(...) declarations.
    return card_row, perf_row, cm_fig, roc_fig, pr_fig, calib_fig_obj, imp_fig, note
