# dash_app/pages/model_performance.py
# PAGE 4 — XAI & Model Performance. Lets a team member FAIRLY compare the four cancellation
# models against the naive historical-average baseline (same estimand, same rows, same
# label — see src.model_eval), inspect where each model beats the baseline, and click into
# a single booking's explanation. Read-only: every figure reads the pre-computed eval /
# SHAP artifacts (Data/model_eval_*.parquet, Data/shap_*.parquet); the heavy compute is
# pre-warmed offline (`python main.py eval` / `python main.py explain`).
#
# Design: same shared system as the other pages (components/ui, theme.brand_figure, dmc
# Stack, chart_card, kpi_strip, location_filter, right-side Drawer). The overbooking COST
# parameter is read from the GLOBAL shared cost-store (single source of truth).

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from dash_app.backend import data_access as da
from dash_app.backend import explain as ex
from dash_app.backend import model_performance as mp
from dash_app.components import performance_charts as pc
from dash_app.components import shap_explain as se
from dash_app.components import ui

dash.register_page(__name__, path="/model-performance", name="Model Performance",
                   order=3, title="STAYERY · Model Performance")

_MODEL_LABELS = {"hazard": "Hazard (survival)", "xgboost": "XGBoost",
                 "histgb": "HistGB", "logreg": "Logistic Regression"}
_METRIC_DATA = [{"label": "AUC", "value": "auc"}, {"label": "Brier", "value": "brier"}]

_INFO = {
    "kpi": "All metrics come from one leak-free, decision-time walk-forward per model "
           "(each booking scored once at d = min(lead, 14) days before arrival, on the same "
           "rows + label). The baseline is the historical average cancel rate.",
    "roc": "ROC on the pooled out-of-time predictions. No baseline line here on purpose: a "
           "constant historical-average predictor sits on the diagonal (AUC 0.5) by "
           "construction, so an AUC-vs-baseline comparison is not meaningful.",
    "roc_loc": "ROC-AUC per location (pooled predictions). Locations with fewer than "
               f"{mp.LOC_MIN_N} bookings or only one class are hidden as unreliable.",
    "pr": "Precision / recall / F1 across the decision threshold. The dashed line is the "
          "cost-optimal threshold for the current walk/empty costs; the orange diamond is "
          "the constant-baseline F1 at that threshold (here the comparison IS valid).",
    "rel": "Reliability diagram: predicted vs observed cancel frequency. On-diagonal = well "
           "calibrated. The orange diamond is the historical-average baseline (a good sanity "
           "check — it should sit on the diagonal at the base rate).",
    "tt": "Train vs test metric, averaged over the walk-forward folds (± std). A large "
          "train-test gap signals overfitting.",
    "iter": "Boosting train/validation loss per iteration — only defined for the boosting "
            "models (XGBoost, HistGB); other model types show nothing rather than a faked curve.",
    "imp": "Mean |SHAP| per feature — model-agnostic, on the same P(cancel-by-arrival) scale "
           "for every model, so importances are comparable across the four.",
    "bee": "SHAP beeswarm: each dot is a booking; colour is the feature value (blue low, red "
           "high). Model-agnostic on the scalar output, so the hazard model is explained on "
           "the same estimand as the classifiers (no SurvSHAP needed).",
    "pdp": "Partial dependence: average P(cancel) as one feature is swept across its range, "
           "with faint per-booking ICE lines. Computed through the same adapter for every model.",
    "table": "Upcoming scored bookings (highest risk first). Click a row for its SHAP "
             "explanation on the selected model.",
}


def _sel(value) -> list[str] | None:
    return list(value) if value else None


def _default_model() -> str:
    for m in mp.registered_models():
        if mp.eval_status(m)["available"]:
            return m
    return mp.registered_models()[0]


# ---------------------------------------------------------------------------
# Layout (callable => artifact availability re-checked on each navigation)
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    props = da.property_list()
    model0 = _default_model()

    controls = dmc.Paper(dmc.Group([
        dmc.Select(id="mp-model", label="Model", value=model0,
                   data=[{"label": _MODEL_LABELS[m], "value": m} for m in mp.registered_models()],
                   allowDeselect=False, style={"width": "220px"}),
        dmc.NumberInput(id="mp-cost-walk", label="Walk cost (€)", value=mp.DEFAULT_WALK,
                        min=0, step=10, style={"width": "150px"}),
        dmc.NumberInput(id="mp-cost-empty", label="Empty-room cost (€)", value=mp.DEFAULT_EMPTY,
                        min=0, step=10, style={"width": "170px"}),
        dmc.Stack([dmc.Text("Cost-optimal threshold", size="xs", c="dimmed", fw=600),
                   dmc.Text(id="mp-thr-badge", size="sm", fw=700)], gap=0),
    ], align="flex-end", gap="md", wrap="wrap"), p="md", radius="lg", withBorder=True)

    tt_extra = dmc.SegmentedControl(id="mp-tt-metric", data=_METRIC_DATA, value="auc",
                                    size="xs", radius="md")
    pdp_extra = dmc.Select(id="mp-pdp-feature", data=[], placeholder="feature",
                           searchable=True, style={"width": "220px"}, size="xs")

    booking_grid = dag.AgGrid(
        id="mp-booking-grid", rowData=[], getRowId="params.data.bid",
        columnDefs=[
            {"field": "bid", "hide": True},
            {"headerName": "Location", "field": "property_name", "flex": 2},
            {"headerName": "Arrival", "field": "arrival_date", "flex": 1},
            {"headerName": "P(cancel)", "field": "cancel_pct", "flex": 1},
            {"headerName": "Risk", "field": "risk_bucket", "flex": 1},
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        columnSize="responsiveSizeToFit", style={"height": "360px"},
        dashGridOptions={"rowSelection": "single", "animateRows": True})

    drawer = dmc.Drawer(id="mp-drawer", position="right", size="lg", padding="lg", opened=False,
                        title=dmc.Text("Booking explanation", fw=700), withCloseButton=True,
                        children=html.Div(id="mp-drawer-body"))

    return dmc.Stack([
        dmc.Group([
            dmc.Group([dmc.Title("Model performance & explainability", order=3),
                       dmc.Badge("Leak-free · same estimand", color="gray", variant="light",
                                 radius="sm")], gap="sm", align="center"),
            dmc.Text("Fair, matched comparison of the four models vs the naive baseline, plus "
                     "SHAP / PDP explanations.", size="sm", c="dimmed"),
        ], justify="space-between", align="center", wrap="wrap", mb="xs"),

        controls,
        ui.location_filter(props, "mp-location-filter"),
        html.Div(id="mp-status"),
        html.Div(id="mp-kpi", children=dmc.Skeleton(height=96, radius="lg")),

        dmc.SimpleGrid([
            ui.chart_card("ROC curve", "mp-roc", info=_INFO["roc"], height=360),
            ui.chart_card("ROC-AUC per location", "mp-roc-loc", info=_INFO["roc_loc"], height=360),
        ], cols={"base": 1, "md": 2}, spacing="md"),

        ui.chart_card("Precision / Recall / F1 · cost-optimal threshold", "mp-pr",
                      info=_INFO["pr"], height=380),
        ui.chart_card("Calibration (reliability)", "mp-reliability", info=_INFO["rel"], height=380),

        dmc.SimpleGrid([
            ui.chart_card("Train vs test", "mp-traintest", info=_INFO["tt"], height=340,
                          header_extra=tt_extra),
            ui.chart_card("Iteration curve", "mp-itercurve", info=_INFO["iter"], height=340),
        ], cols={"base": 1, "md": 2}, spacing="md"),

        dmc.SimpleGrid([
            ui.chart_card("Feature importance (mean |SHAP|)", "mp-importance", info=_INFO["imp"],
                          height=440),
            ui.chart_card("SHAP beeswarm", "mp-beeswarm", info=_INFO["bee"], height=460),
        ], cols={"base": 1, "md": 2}, spacing="md"),

        ui.chart_card("Partial dependence / ICE", "mp-pdp", info=_INFO["pdp"], height=360,
                      header_extra=pdp_extra),

        dmc.Card([
            dmc.Group([dmc.Text("Single-booking explanations", fw=600, size="sm"),
                       ui.info_icon(_INFO["table"])], gap=6),
            dmc.Space(h=6), booking_grid,
        ], withBorder=True, radius="lg", p="md", shadow="xs"),

        drawer,
    ], gap="md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(x, d=1):
    return "unavailable" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.{d}f}%"


def _num(x, d=3):
    return "unavailable" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def _status_alert(model: str):
    st = mp.eval_status(model)
    if not st["available"]:
        return dmc.Alert(
            dmc.Text(f"No evaluation artifact for '{model}' yet. Run "
                     f"`python main.py eval --model {model}` (once, offline) to populate the "
                     "charts below.", size="sm"),
            title="Evaluation not built", color="yellow", variant="light",
            icon=html.I(className="bi bi-exclamation-triangle"), radius="md")
    if not ex.shap_available(model):
        return dmc.Alert(
            dmc.Text(f"Metrics are ready. SHAP/importance for '{model}' not built yet — run "
                     f"`python main.py explain --model {model}` to fill the SHAP charts.",
                     size="sm"),
            title="SHAP not built", color="blue", variant="light",
            icon=html.I(className="bi bi-info-circle"), radius="md", withCloseButton=True)
    return None


def _kpi_cards(k: dict) -> list:
    beats = (k["bss"] is not None and k["bss"] > 0)
    return [
        ui.kpi_card("Discrimination (AUC)", _num(k["auc"]),
                    sub=f"{k['n']:,} pooled bookings" if k["n"] else "no data",
                    accent=True, tooltip="Pooled ROC-AUC on the leak-free walk-forward."),
        ui.kpi_card("Brier skill vs baseline", _num(k["bss"]),
                    sub=("beats baseline" if beats else "not beating baseline"),
                    tooltip="Brier Skill Score vs the historical-average predictor. >0 = better "
                            "than always guessing the base rate."),
        ui.kpi_card("Best location", k["best_loc"] or "unavailable",
                    sub=f"BSS {_num(k['best_bss'])}" if k["best_loc"] else "below sample threshold",
                    tooltip="Location where the model most beats its baseline (Brier Skill)."),
        ui.kpi_card("Weakest location", k["worst_loc"] or "unavailable",
                    sub=f"BSS {_num(k['worst_bss'])}" if k["worst_loc"] else "below sample threshold",
                    tooltip="Location where the model least beats (or trails) its baseline."),
    ]


# ---------------------------------------------------------------------------
# Prefill cost inputs from the shared store (single source of truth); resync on model change
# ---------------------------------------------------------------------------
@callback(
    Output("mp-cost-walk", "value"),
    Output("mp-cost-empty", "value"),
    Input("mp-model", "value"),
    State("cost-store", "data"),
)
def _prefill_cost(_model, store):
    walk, empty = mp.read_cost_params(store)
    return walk, empty


@callback(
    Output("cost-store", "data", allow_duplicate=True),
    Input("mp-cost-walk", "value"),
    Input("mp-cost-empty", "value"),
    State("cost-store", "data"),
    prevent_initial_call=True,
)
def _save_cost(walk, empty, store):
    store = dict(store or {})
    cur = dict(store.get(mp.GLOBAL_COST_KEY) or {})
    cur.update({"walk": walk, "empty": empty})
    store[mp.GLOBAL_COST_KEY] = cur
    return store


# ---------------------------------------------------------------------------
# Core metrics: KPI + ROC + PR + reliability (react to model / location / cost)
# ---------------------------------------------------------------------------
@callback(
    Output("mp-status", "children"),
    Output("mp-kpi", "children"),
    Output("mp-roc", "figure"),
    Output("mp-roc-loc", "figure"),
    Output("mp-pr", "figure"),
    Output("mp-reliability", "figure"),
    Output("mp-thr-badge", "children"),
    Input("mp-model", "value"),
    Input("mp-location-filter", "value"),
    Input("mp-cost-walk", "value"),
    Input("mp-cost-empty", "value"),
)
def _update_core(model, sel_value, walk, empty):
    props = _sel(sel_value)
    walk = float(walk) if walk not in (None, "") else mp.DEFAULT_WALK
    empty = float(empty) if empty not in (None, "") else mp.DEFAULT_EMPTY

    status = _status_alert(model)
    k = mp.kpis(model, props, walk, empty)
    kpi = ui.kpi_strip(_kpi_cards(k)) if k.get("n") else dmc.Alert(
        "No evaluation data for this selection yet.", color="gray", variant="light")

    pr = mp.pr_threshold(model, props, walk, empty)
    thr_txt = f"{pr['t_cost']:.2f}" if pr else "—"
    return (status, kpi,
            pc.fig_roc(mp.roc_global(model, props)),
            pc.fig_roc_by_location(mp.roc_by_location(model, props)),
            pc.fig_pr_threshold(pr),
            pc.fig_reliability(mp.reliability(model, props)),
            thr_txt)


# ---------------------------------------------------------------------------
# Train vs test + iteration curve
# ---------------------------------------------------------------------------
@callback(
    Output("mp-traintest", "figure"),
    Output("mp-itercurve", "figure"),
    Input("mp-model", "value"),
    Input("mp-tt-metric", "value"),
)
def _update_traintest(model, metric):
    tt = mp.train_test(model)
    try:
        curve = ex.iteration_curve(model)
    except Exception:  # noqa: BLE001
        curve = {}
    return pc.fig_train_test(tt, metric=metric or "auc"), pc.fig_iteration_curve(curve)


# ---------------------------------------------------------------------------
# SHAP importance + beeswarm (cost-independent)
# ---------------------------------------------------------------------------
@callback(
    Output("mp-importance", "figure"),
    Output("mp-beeswarm", "figure"),
    Input("mp-model", "value"),
)
def _update_xai(model):
    return pc.fig_importance(ex.importance_from_shap(model)), pc.fig_beeswarm(ex.global_beeswarm(model))


# ---------------------------------------------------------------------------
# PDP: feature options + curve
# ---------------------------------------------------------------------------
@callback(
    Output("mp-pdp-feature", "data"),
    Output("mp-pdp-feature", "value"),
    Input("mp-model", "value"),
)
def _pdp_options(model):
    feats = ex.explainable_features(model)
    if not feats:
        return [], None
    return [{"label": f, "value": f} for f in feats], feats[0]


@callback(
    Output("mp-pdp", "figure"),
    Input("mp-model", "value"),
    Input("mp-pdp-feature", "value"),
)
def _update_pdp(model, feature):
    if not feature:
        return pc.fig_pdp({})
    try:
        return pc.fig_pdp(ex.partial_dependence(model, feature))
    except Exception:  # noqa: BLE001
        return pc.fig_pdp({})


# ---------------------------------------------------------------------------
# Single-booking table + drawer explanation (spec 4.8)
# ---------------------------------------------------------------------------
@callback(
    Output("mp-booking-grid", "rowData"),
    Input("mp-location-filter", "value"),
)
def _fill_table(sel_value):
    df = da.load_scored()
    if df.empty:
        return []
    df = df.reset_index(drop=True)
    df["bid"] = df.index.astype(str)
    props = _sel(sel_value)
    if props and "property_name" in df.columns:
        df = df[df["property_name"].isin(props)]
    df = df.sort_values("cancel_proba", ascending=False).head(200)
    rows = []
    for r in df.itertuples():
        arr = getattr(r, "arrival", None)
        rows.append({
            "bid": r.bid,
            "property_name": getattr(r, "property_name", "—"),
            "arrival_date": pd.to_datetime(arr, utc=True, errors="coerce").strftime("%Y-%m-%d")
                            if arr is not None else "—",
            "cancel_pct": f"{float(getattr(r, 'cancel_proba', 0)) * 100:.1f}%",
            "risk_bucket": getattr(r, "risk_bucket", "—"),
        })
    return rows


@callback(
    Output("mp-drawer", "opened"),
    Output("mp-drawer", "title"),
    Output("mp-drawer-body", "children"),
    Input("mp-booking-grid", "cellClicked"),
    State("mp-model", "value"),
    prevent_initial_call=True,
)
def _explain_booking(cell, model):
    if not cell:
        return no_update, no_update, no_update
    df = da.load_scored().reset_index(drop=True)
    try:
        bid = int(cell["rowId"])
        booking = df.iloc[bid]
    except Exception:  # noqa: BLE001
        return no_update, no_update, no_update
    title = dmc.Text(f"{booking.get('property_name', 'Booking')} · "
                     f"P(cancel) {float(booking.get('cancel_proba', 0)) * 100:.1f}%", fw=700)
    return True, title, se.explanation_panel(model, booking, mini=False)
