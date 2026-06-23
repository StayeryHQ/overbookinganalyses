# dash_app/pages/overbooking_predictions.py
# ---------------------------------------------------------------------------
# "Overbooking Predictions" — DAY-LEVEL + interactive.
#
# The recommendation is per (Standort, Tag) via derive.recommendation_by_day,
# factoring the day's changing occupancy AND its cancellation risk. The user
# picks a day (sidebar) OR clicks a heatmap cell; the table then shows every
# location's recommended overbookings FOR THAT DAY.
#
# Filters (model + day + decision-threshold slider) live in the SIDEBAR via the
# per-page filters_registry.
# ---------------------------------------------------------------------------

from __future__ import annotations

import dash
import pandas as pd
from dash import callback, dcc, html, dash_table, Input, Output, ctx

from dash_app import backend as B
from dash_app.backend import derive as D
from dash_app.backend import occupancy as occ          # property_performance (occupancy)
from dash_app.backend import schema as S
from dash_app.components import explain, hero, metric_row, section
from dash_app.filters_registry import register
from dash_app import config as CFG
from src.plotting import bars, risk_heatmap

dash.register_page(
    __name__,
    path="/overbooking-predictions",
    name="Overbooking Predictions",
    title="Overbooking Predictions · Stayery",
    order=CFG.PAGE_ORDER["overbooking_predictions"],
)

# Element ids.
_MODEL_DD, _DAY_DD = "pred-model-dropdown", "pred-day-dropdown"
_THR_SLIDER = "pred-thr-slider"                           # manual decision-threshold slider
_KPI, _HEATMAP, _BUCKET = "pred-kpi", "pred-heatmap", "pred-bucket"
_REC_TABLE, _DAY_INFO = "pred-rec-table", "pred-day-info"

HORIZON = 14                                              # forecast window length (days)

# Slider start value = the auto-picked model's cost-optimal validation threshold
# (falls back to the analytic Bayes value if no model/src is available).
try:
    _DEFAULT_THR = round(float(B.default_threshold(None)), 2)
except Exception:  # noqa: BLE001 — never let a bad default break page import
    _DEFAULT_THR = 0.79


def _fmt(d) -> str:
    """Day label used identically on the heatmap columns + day dropdown + table."""
    return pd.Timestamp(d).strftime("%a %d.%m.")


def _window_dates() -> list[pd.Timestamp]:
    """The next HORIZON days from today (normalised)."""
    today = pd.Timestamp.today().normalize()
    return [today + pd.Timedelta(days=i) for i in range(HORIZON)]


def _model_options() -> list[dict]:
    """Dropdown options: 'auto' + each available/known model."""
    opts = [{"label": "Automatisch (bester AUC)", "value": ""}]
    avail = B.available_models()
    for name in (avail if avail else list(CFG.MODEL_LABELS.keys())):
        opts.append({"label": CFG.MODEL_LABELS.get(name, name), "value": name})
    return opts


# =============================================================================
# Sidebar filters (model + day + threshold) — registered for this page
# =============================================================================
def filters_layout() -> list:
    """Model + day + decision-threshold selectors, rendered into the sidebar slot."""
    day_opts = [{"label": _fmt(d), "value": _fmt(d)} for d in _window_dates()]
    return [
        html.Div("Modell", className="stayery-sidebar-heading"),
        # Model selector — swaps the scoring model with no other change.
        dcc.Dropdown(id=_MODEL_DD, options=_model_options(), value="", clearable=False,
                     persistence=True),
        html.Div("Tag", className="stayery-control-label"),
        # Day selector — the table shows all locations' recommendations for it.
        dcc.Dropdown(id=_DAY_DD, options=day_opts, value=day_opts[0]["value"],
                     clearable=False, persistence=True),
        html.Div("Entscheidungs-Schwelle", className="stayery-control-label"),
        # Manual decision threshold. Starts at the model's cost-optimal validation
        # value; analysts drag it to be more (right) or less (left) conservative.
        dcc.Slider(id=_THR_SLIDER, min=0.0, max=1.0, step=0.01, value=_DEFAULT_THR,
                   marks={0.0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1"},
                   tooltip={"placement": "bottom", "always_visible": True},
                   persistence=False),
        html.Div(f"Standard ≈ {_DEFAULT_THR:.2f} = kostenoptimal auf Validierung "
                 "(Kosten: Walk 300 € vs. leeres Zimmer 80 €). Höher = konservativer: "
                 "nur sehr sichere Stornos zählen als Overbooking-Spielraum.",
                 className="stayery-caption"),
    ]


register("/overbooking-predictions", filters_layout)


# =============================================================================
# Compute
# =============================================================================
def _compute(model_value: str, threshold: float):
    """Return (kpi, heatmap_fig, bucket_fig, rec_by_day_frame) for a model + threshold."""
    B.set_model(model_value or None)                     # "" -> auto-pick by AUC
    bookings = B.get_scored_bookings()
    units = B.units_by_hotel()
    labels = B.hotel_labels()
    dates = _window_dates()
    perf = occ.get_perf()                                # occupancy from property_performance

    # Per-(Standort, Tag) recommendation: occupancy gates, the CONSERVATIVE count
    # of high-confidence cancellations (p >= threshold) sets the amount.
    rec = D.recommendation_by_day(bookings, dates, units, perf=perf, labels=labels,
                                  threshold=threshold)

    # KPI row over the whole window.
    conf = D.confirmed(bookings)
    win = conf[conf[S.ARRIVAL_DATE].isin(dates)]
    n_sure = int((win[S.CANCEL_PROBA] >= threshold).sum())
    kpi = metric_row([
        {"label": "Anreisen im Fenster", "value": f"{len(win):,}".replace(",", ".")},
        {"label": "Erwartete Stornos (Σp)", "value": f"{win[S.CANCEL_PROBA].sum():.0f}",
         "help": "Σ Storno-Wahrscheinlichkeiten über das Fenster (risikoneutraler Erwartungswert)."},
        {"label": f"Sichere Stornos (p ≥ {threshold:.2f})", "value": f"{n_sure:,}".replace(",", "."),
         "help": "Buchungen, bei denen das Modell hinreichend sicher ist (konservativ)."},
        {"label": "Σ Empfehlung (Fenster)", "value": int(rec["Empfehlung"].sum())},
    ])

    # Recommendation heatmap: location × day, coloured by recommended overbookings.
    pm = rec.pivot_table(index="Standort", columns="Datum", values="Empfehlung", aggfunc="first")
    pm = pm.reindex(columns=[_fmt(d) for d in dates])     # chronological day order
    heatmap_fig = risk_heatmap(pm, title="Empfohlene Overbookings je Standort & Tag",
                               colorbar_title="Empf.")

    # Risk-bucket distribution, re-derived live from the slider: high = p >= threshold,
    # low = below the validation base rate (below-average risk), uncertain = between.
    low_b, _ = B.serving_bounds(model_value or None)
    low_b = min(low_b, threshold)                         # keep low <= high
    p = win[S.CANCEL_PROBA]
    counts = {"high": int((p >= threshold).sum()),
              "uncertain": int(((p >= low_b) & (p < threshold)).sum()),
              "low": int((p < low_b).sum())}
    bucket_fig = bars([S.RISK_LABELS[b] for b in S.RISK_BUCKETS],
                      [counts[b] for b in S.RISK_BUCKETS],
                      title="Buchungen je Risiko-Stufe (Schwelle interaktiv)",
                      yaxis_title="Buchungen", color_name="yellow", text_fmt=".0f")
    return kpi, heatmap_fig, bucket_fig, rec


def layout(**kwargs) -> html.Div:
    """Page content (model + day + threshold selectors live in the sidebar)."""
    return html.Div([
        hero(eyebrow="Vorhersage", title="Overbooking Predictions",
             subtitle="Empfohlene Overbookings je Standort und Tag — Auslastung + Storno-"
                      "Wahrscheinlichkeit pro Tag, nicht über das Fenster gemittelt."),
        explain("Wie entsteht die Empfehlung?",
                "Pro Standort UND Tag: Empfehlung = 0 bei Slack (Auslastung < 85 %), sonst die "
                "Anzahl SICHERER Stornos des Tages (Storno-Wahrscheinlichkeit ≥ Schwelle), "
                "gedeckelt aufs Limit (< 50 Units → 2, ≥ 50 → 4). Die Schwelle links in der "
                "Sidebar steuert, wie konservativ wir sind (Standard = kostenoptimal auf "
                "Validierung). Tag wählen ODER eine Heatmap-Zelle anklicken — die Tabelle "
                "filtert automatisch mit."),
        html.Div(id=_KPI),
        section(1, "Empfehlungs-Heatmap (Standort × Tag)",
                description="Farbe = empfohlene Overbookings. Klick auf eine Zelle wählt den Tag.",
                children=[dcc.Graph(id=_HEATMAP, config={"displayModeBar": False})]),
        section(2, "Empfehlung für den gewählten Tag",
                description="Je Standort: Auslastung, Anreisen, erwartete Stornos, freie Zimmer, "
                            "Limit und Empfehlung.",
                children=[
                    html.Div(id=_DAY_INFO, className="stayery-caption"),
                    dash_table.DataTable(
                        id=_REC_TABLE, sort_action="native", page_size=14,
                        style_as_list_view=True,
                        style_header={"fontWeight": "700", "backgroundColor": "#FFF7CC"},
                        style_cell={"fontFamily": "Inter, Helvetica, Arial, sans-serif",
                                    "padding": "6px 10px", "textAlign": "left", "fontSize": "13px"},
                        style_table={"overflowX": "auto"},
                    ),
                ]),
        section(3, "Risiko-Verteilung",
                description="Wie viele Buchungen in jeder Risiko-Stufe liegen.",
                children=[dcc.Graph(id=_BUCKET, config={"displayModeBar": False})]),
    ])


# =============================================================================
# Callback: model OR day OR heatmap-click OR threshold -> recompute + filter day table
# =============================================================================
@callback(
    Output(_KPI, "children"), Output(_HEATMAP, "figure"), Output(_BUCKET, "figure"),
    Output(_REC_TABLE, "data"), Output(_REC_TABLE, "columns"), Output(_DAY_INFO, "children"),
    Input(_MODEL_DD, "value"), Input(_DAY_DD, "value"), Input(_HEATMAP, "clickData"),
    Input(_THR_SLIDER, "value"),
)
def _update(model_value, day_value, click, threshold):
    """Recompute everything; the table shows all locations for the chosen day."""
    thr = float(threshold) if threshold is not None else _DEFAULT_THR
    kpi, heatmap_fig, bucket_fig, rec = _compute(model_value or "", thr)

    # The chosen day = clicked cell's day (if the heatmap fired) else the dropdown.
    sel_day = day_value
    if ctx.triggered_id == _HEATMAP and click:
        sel_day = click["points"][0]["x"]                # heatmap x label == _fmt(date)

    # Table: every location's recommendation for the chosen day, strongest first.
    day_rec = rec[rec["Datum"] == sel_day].drop(columns=["Datum"])
    day_rec = day_rec.sort_values("Empfehlung", ascending=False)
    info = (f"Gewählter Tag: {sel_day} — {int(day_rec['Empfehlung'].sum())} empfohlene "
            f"Overbookings über {len(day_rec)} Standorte.")
    columns = [{"name": c, "id": c} for c in day_rec.columns]
    return kpi, heatmap_fig, bucket_fig, day_rec.to_dict("records"), columns, info


# =============================================================================
# Callback: model change -> reset the threshold slider to THAT model's
# cost-optimal validation default (analysts can then drag from there).
# =============================================================================
@callback(
    Output(_THR_SLIDER, "value"),
    Input(_MODEL_DD, "value"),
)
def _reset_threshold(model_value):
    """Set the slider to the selected model's cost-optimal validation threshold."""
    try:
        return round(float(B.default_threshold(model_value or None)), 2)
    except Exception:  # noqa: BLE001
        return _DEFAULT_THR
